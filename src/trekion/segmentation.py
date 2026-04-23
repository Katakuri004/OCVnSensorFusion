from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class Phase3Config:
    model_id: str = "yolov8s-seg.pt"
    device: str = "auto"
    conf: float = 0.25
    imgsz: int = 640
    enable_hands: bool = True
    enable_depth: bool = False
    enable_hud: bool = True
    class_whitelist: set[str] = field(
        default_factory=lambda: {"person", "book", "laptop", "tv", "cell phone", "backpack"}
    )
    class_remap: dict[str, str] = field(
        default_factory=lambda: {
            "tv": "display",
            "laptop": "display",
            "book": "notebook",
            "cell phone": "device",
            "backpack": "bag",
            "person": "person",
            "hand": "hand",
        }
    )
    per_class_conf: dict[str, float] = field(
        default_factory=lambda: {
            "display": 0.25,
            "notebook": 0.30,
            "device": 0.30,
            "bag": 0.30,
            "person": 0.30,
            "hand": 0.25,
        }
    )
    min_box_area_ratio: float = 0.002
    class_area_range_ratio: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "display": (0.03, 0.95),
            "notebook": (0.01, 0.40),
            "device": (0.002, 0.12),
            "bag": (0.01, 0.40),
            "person": (0.03, 0.85),
            "hand": (0.002, 0.20),
        }
    )
    class_aspect_range: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "display": (0.7, 4.0),
            "notebook": (0.6, 2.4),
            "device": (0.3, 2.2),
            "bag": (0.5, 2.3),
            "person": (0.2, 1.2),
            "hand": (0.4, 2.2),
        }
    )
    nms_iou_threshold: float = 0.45
    mask_open_kernel: int = 3
    mask_close_kernel: int = 3
    edge_ignore_ratio: float = 0.0
    min_confirm_frames: int = 2
    track_ttl_frames: int = 4
    smoothing_alpha: float = 0.6
    class_persistence_frames: int = 1
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_spread_limit: float = 0.35
    depth_band_ratio: float = 0.2
    depth_refine_classes: set[str] = field(default_factory=lambda: {"display"})
    depth_reject_by_spread: bool = False
    motion_gyro_threshold: float = 4.0
    motion_confidence_boost: float = 0.0
    high_motion_extra_confirm_frames: int = 0
    track_center_dist_ratio: float = 0.4
    track_size_ratio_range: tuple[float, float] = (0.4, 2.5)
    apply_scene_priors_pretrack: bool = False
    apply_scene_priors_posttrack: bool = False
    hand_model_cache_dir: Path = Path("models")


@dataclass
class SceneDetection:
    raw_class_name: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    track_id: int | None = None
    depth_median: float | None = None
    depth_spread: float | None = None


@dataclass
class HandDetection:
    landmarks_norm: list[tuple[float, float]]
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    stable: bool = False


@dataclass
class TrackState:
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    class_name: str
    hits: int = 1
    missed: int = 0
    label_votes: dict[str, int] = field(default_factory=dict)


@dataclass
class FrameDiagnostics:
    frame_index: int
    timestamp_ms: int
    inference_ms: float
    active_detections: int
    hands_count: int
    high_motion: bool
    model_name: str
    device: str
    stage_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class FusedFrame:
    detections: list[SceneDetection]
    hands: list[HandDetection]
    diagnostics: FrameDiagnostics


@dataclass
class Phase3Pipeline:
    config: Phase3Config
    detector: Any
    class_names: dict[int, str]
    hand_landmarker: mp.tasks.vision.HandLandmarker | None = None
    depth_pipe: Any | None = None
    tracks: dict[int, TrackState] = field(default_factory=dict)
    next_track_id: int = 1
    class_memory: dict[str, int] = field(default_factory=dict)


def yolo_predict_device(flag: str) -> str | int:
    if flag == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if flag == "cpu":
        return "cpu"
    return 0


def ensure_hand_landmarker_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "hand_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def create_hand_landmarker(model_path: Path) -> mp.tasks.vision.HandLandmarker:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def segmentation_pipeline(config: Phase3Config) -> Phase3Pipeline:
    from ultralytics import YOLO

    detector = YOLO(config.model_id)
    class_names = detector.names if isinstance(detector.names, dict) else {}

    hand_landmarker = None
    if config.enable_hands:
        model_path = ensure_hand_landmarker_model(config.hand_model_cache_dir)
        hand_landmarker = create_hand_landmarker(model_path)

    depth_pipe = None
    if config.enable_depth:
        from trekion.depth import depth_pipeline

        depth_pipe = depth_pipeline(config.depth_model_id, config.device)

    return Phase3Pipeline(
        config=config,
        detector=detector,
        class_names=class_names,
        hand_landmarker=hand_landmarker,
        depth_pipe=depth_pipe,
    )


def infer_detections(
    pipeline: Phase3Pipeline, frame_bgr: np.ndarray
) -> tuple[list[SceneDetection], float, dict[str, int]]:
    t0 = perf_counter()
    results = pipeline.detector.predict(
        source=frame_bgr,
        conf=pipeline.config.conf,
        imgsz=pipeline.config.imgsz,
        device=yolo_predict_device(pipeline.config.device),
        verbose=False,
    )
    inference_ms = (perf_counter() - t0) * 1000.0
    result = results[0]

    boxes = _tensor_to_numpy(getattr(result.boxes, "xyxy", None))
    confs = _tensor_to_numpy(getattr(result.boxes, "conf", None))
    classes = _tensor_to_numpy(getattr(result.boxes, "cls", None))
    masks = _extract_masks(result)

    detections: list[SceneDetection] = []
    stage_counts = {
        "raw": 0,
        "after_whitelist": 0,
        "after_conf": 0,
        "after_min_area": 0,
        "after_scene_priors": 0,
        "after_nms": 0,
    }
    if boxes is None or confs is None or classes is None:
        return detections, inference_ms, stage_counts

    h, w = frame_bgr.shape[:2]
    min_area_px = pipeline.config.min_box_area_ratio * float(h * w)
    stage_counts["raw"] = len(boxes)
    for i in range(len(boxes)):
        raw_name = pipeline.class_names.get(int(classes[i]), str(int(classes[i])))
        if pipeline.config.class_whitelist and raw_name not in pipeline.config.class_whitelist:
            continue
        stage_counts["after_whitelist"] += 1
        class_name = pipeline.config.class_remap.get(raw_name, raw_name)
        confidence = float(confs[i])
        class_conf = pipeline.config.per_class_conf.get(class_name, pipeline.config.conf)
        if confidence < class_conf:
            continue
        stage_counts["after_conf"] += 1
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < min_area_px:
            continue
        stage_counts["after_min_area"] += 1
        if pipeline.config.apply_scene_priors_pretrack and not _passes_scene_priors(
            class_name=raw_name,
            bbox_xyxy=(x1, y1, x2, y2),
            frame_w=w,
            frame_h=h,
            config=pipeline.config,
        ):
            continue
        stage_counts["after_scene_priors"] += 1
        mask = masks[i] if i < len(masks) else None
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            mask = _cleanup_mask(mask, pipeline.config.mask_open_kernel, pipeline.config.mask_close_kernel)
        detections.append(
            SceneDetection(
                raw_class_name=raw_name,
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=(x1, y1, x2, y2),
                mask=mask,
            )
        )

    detections = _apply_nms(detections, pipeline.config.nms_iou_threshold)
    stage_counts["after_nms"] = len(detections)
    return detections, inference_ms, stage_counts


def infer_hands(pipeline: Phase3Pipeline, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
    if pipeline.hand_landmarker is None:
        return []
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = pipeline.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    hands: list[HandDetection] = []
    for idx, hand in enumerate(result.hand_landmarks):
        xs = [lm.x for lm in hand]
        ys = [lm.y for lm in hand]
        score = 0.0
        if idx < len(result.handedness) and len(result.handedness[idx]) > 0:
            score = float(result.handedness[idx][0].score)
        hands.append(
            HandDetection(
                landmarks_norm=[(float(lm.x), float(lm.y)) for lm in hand],
                bbox_xyxy=(max(0.0, min(xs)) * w, max(0.0, min(ys)) * h, min(1.0, max(xs)) * w, min(1.0, max(ys)) * h),
                confidence=score,
            )
        )
    return hands


def fuse_scene_and_hands(
    pipeline: Phase3Pipeline,
    detections: list[SceneDetection],
    hands: list[HandDetection],
    frame_bgr: np.ndarray,
    frame_index: int,
    timestamp_ms: int,
    imu_row: dict[str, float] | None = None,
    stage_counts: dict[str, int] | None = None,
    inference_ms: float = 0.0,
) -> FusedFrame:
    depth_map = None
    if pipeline.depth_pipe is not None:
        from trekion.depth import infer_depth_map

        depth_map = infer_depth_map(pipeline.depth_pipe, frame_bgr)
    high_motion = _is_high_motion(imu_row, pipeline.config.motion_gyro_threshold)
    stage = dict(stage_counts or {})
    motion_floor = pipeline.config.motion_confidence_boost if high_motion else 0.0
    if depth_map is not None:
        detections = _apply_depth_refinement(detections, depth_map, pipeline.config)
    stage["after_depth"] = len(detections)
    detections = [d for d in detections if d.confidence >= pipeline.config.per_class_conf.get(d.class_name, pipeline.config.conf) + motion_floor]
    stage["after_motion_conf"] = len(detections)
    required_hits = pipeline.config.min_confirm_frames + (pipeline.config.high_motion_extra_confirm_frames if high_motion else 0)
    tracked = _update_tracks(detections, pipeline, required_hits=required_hits)
    stage["after_tracking"] = len(tracked)
    if pipeline.config.apply_scene_priors_posttrack:
        tracked = [
            d
            for d in tracked
            if _passes_scene_priors(
                class_name=d.raw_class_name,
                bbox_xyxy=d.bbox_xyxy,
                frame_w=frame_bgr.shape[1],
                frame_h=frame_bgr.shape[0],
                config=pipeline.config,
            )
        ]
    stage["after_posttrack_priors"] = len(tracked)
    tracked = [
        d
        for d in tracked
        if _away_from_edges(d.bbox_xyxy, frame_bgr.shape[1], frame_bgr.shape[0], pipeline.config.edge_ignore_ratio)
    ]
    stage["after_edge"] = len(tracked)

    for hand in hands:
        hand.stable = hand.confidence >= pipeline.config.per_class_conf.get("hand", 0.2)
    stable_hands = [h for h in hands if h.stable]
    diagnostics = FrameDiagnostics(
        frame_index=frame_index,
        timestamp_ms=timestamp_ms,
        inference_ms=inference_ms,
        active_detections=len(tracked),
        hands_count=len(stable_hands),
        high_motion=high_motion,
        model_name=pipeline.config.model_id,
        device=pipeline.config.device,
        stage_counts=stage,
    )
    return FusedFrame(detections=tracked, hands=stable_hands, diagnostics=diagnostics)


def render_detections(frame_bgr: np.ndarray, fused: FusedFrame) -> np.ndarray:
    out = frame_bgr.copy()
    for det in fused.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = _color_for_label(det.class_name)
        if det.mask is not None:
            overlay = out.copy()
            overlay[det.mask > 0] = (0.65 * np.array(color) + 0.35 * overlay[det.mask > 0]).astype(np.uint8)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for hand in fused.hands:
        points = [(int(x * out.shape[1]), int(y * out.shape[0])) for x, y in hand.landmarks_norm]
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(out, points[a], points[b], (255, 200, 0), 2)
        for px, py in points:
            cv2.circle(out, (px, py), 2, (0, 255, 255), -1)

    d = fused.diagnostics
    if d.model_name:
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (930, 168), (0, 0, 0), -1)
        out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0.0)
        lines = [
            f"frame={d.frame_index} ts_ms={d.timestamp_ms} infer_ms={d.inference_ms:.1f}",
            f"detections={d.active_detections} hands={d.hands_count} high_motion={d.high_motion}",
            f"model={d.model_name} device={d.device}",
        ]
        if d.stage_counts:
            lines.append(
                "stages raw={raw} wl={after_whitelist} conf={after_conf} nms={after_nms} trk={after_tracking} out={after_edge}".format(
                    raw=d.stage_counts.get("raw", 0),
                    after_whitelist=d.stage_counts.get("after_whitelist", 0),
                    after_conf=d.stage_counts.get("after_conf", 0),
                    after_nms=d.stage_counts.get("after_nms", 0),
                    after_tracking=d.stage_counts.get("after_tracking", 0),
                    after_edge=d.stage_counts.get("after_edge", 0),
                )
            )
        y = 34
        for line in lines:
            cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            y += 30
    return out


def draw_hands_bgr(frame_bgr: np.ndarray, landmarker: mp.tasks.vision.HandLandmarker, timestamp_ms: int) -> np.ndarray:
    tmp = Phase3Pipeline(config=Phase3Config(enable_hands=True), detector=None, class_names={}, hand_landmarker=landmarker)
    hands = infer_hands(tmp, frame_bgr, timestamp_ms)
    fused = FusedFrame(
        detections=[],
        hands=hands,
        diagnostics=FrameDiagnostics(
            frame_index=0,
            timestamp_ms=timestamp_ms,
            inference_ms=0.0,
            active_detections=0,
            hands_count=len(hands),
            high_motion=False,
            model_name="",
            device="",
        ),
    )
    return render_detections(frame_bgr, fused)


def _apply_depth_refinement(
    detections: list[SceneDetection],
    depth_map: np.ndarray,
    config: Phase3Config,
) -> list[SceneDetection]:
    refined: list[SceneDetection] = []
    lo = float(np.percentile(depth_map, 5))
    hi = float(np.percentile(depth_map, 95))
    scale = max(hi - lo, 1e-6)
    for det in detections:
        if det.class_name not in config.depth_refine_classes:
            refined.append(det)
            continue
        if det.mask is not None and det.mask.any():
            vals = depth_map[det.mask > 0]
        else:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            vals = depth_map[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)].reshape(-1)
        if vals.size == 0:
            continue
        median = float(np.median(vals))
        spread = float(np.std(vals) / scale)
        if config.depth_reject_by_spread and spread > config.depth_spread_limit:
            continue
        det.depth_median = median
        det.depth_spread = spread
        if det.mask is not None:
            band = config.depth_band_ratio * scale
            keep = np.abs(depth_map - median) <= band
            det.mask = np.logical_and(det.mask > 0, keep).astype(np.uint8)
        refined.append(det)
    refined.sort(key=lambda d: d.depth_median if d.depth_median is not None else float("inf"))
    return refined


def _update_tracks(
    detections: list[SceneDetection],
    pipeline: Phase3Pipeline,
    required_hits: int | None = None,
) -> list[SceneDetection]:
    if required_hits is None:
        required_hits = pipeline.config.min_confirm_frames
    assigned_tracks: set[int] = set()
    for det in detections:
        best_track = None
        best_iou = 0.0
        for tid, track in pipeline.tracks.items():
            if tid in assigned_tracks:
                continue
            if track.class_name != det.class_name:
                continue
            if not _track_shape_center_compatible(track.bbox_xyxy, det.bbox_xyxy, pipeline.config):
                continue
            iou = _bbox_iou(det.bbox_xyxy, track.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is not None and best_iou >= 0.2:
            best_track.bbox_xyxy = _smooth_bbox(best_track.bbox_xyxy, det.bbox_xyxy, pipeline.config.smoothing_alpha)
            best_track.missed = 0
            best_track.hits += 1
            best_track.label_votes[det.class_name] = best_track.label_votes.get(det.class_name, 0) + 1
            best_track.class_name = max(best_track.label_votes, key=best_track.label_votes.get)
            det.bbox_xyxy = best_track.bbox_xyxy
            det.class_name = best_track.class_name
            det.track_id = best_track.track_id
            assigned_tracks.add(best_track.track_id)
        else:
            tid = pipeline.next_track_id
            pipeline.next_track_id += 1
            pipeline.tracks[tid] = TrackState(
                track_id=tid,
                bbox_xyxy=det.bbox_xyxy,
                class_name=det.class_name,
                label_votes={det.class_name: 1},
            )
            det.track_id = tid
            assigned_tracks.add(tid)

    for tid, track in list(pipeline.tracks.items()):
        if tid not in assigned_tracks:
            track.missed += 1
        if track.missed > pipeline.config.track_ttl_frames:
            del pipeline.tracks[tid]

    out: list[SceneDetection] = []
    for det in detections:
        if det.track_id is None:
            continue
        track = pipeline.tracks.get(det.track_id)
        if track is None:
            continue
        if track.hits >= required_hits:
            out.append(det)
    return out


def _tensor_to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_masks(result: Any) -> list[np.ndarray]:
    if not hasattr(result, "masks") or result.masks is None:
        return []
    data = getattr(result.masks, "data", None)
    arr = _tensor_to_numpy(data)
    if arr is None:
        return []
    return [(m > 0.5).astype(np.uint8) for m in arr]


def _cleanup_mask(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    out = (mask > 0).astype(np.uint8)
    if open_kernel > 1:
        k = np.ones((open_kernel, open_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def _apply_nms(detections: list[SceneDetection], iou_threshold: float) -> list[SceneDetection]:
    kept: list[SceneDetection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(
            _bbox_iou(det.bbox_xyxy, other.bbox_xyxy) < iou_threshold
            for other in kept
            if other.raw_class_name == det.raw_class_name
        ):
            kept.append(det)
    return kept


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _smooth_bbox(
    prev_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    p = np.asarray(prev_bbox, dtype=np.float32)
    n = np.asarray(new_bbox, dtype=np.float32)
    s = alpha * p + (1.0 - alpha) * n
    return float(s[0]), float(s[1]), float(s[2]), float(s[3])


def _away_from_edges(bbox: tuple[float, float, float, float], w: int, h: int, edge_ratio: float) -> bool:
    margin_w = edge_ratio * w
    margin_h = edge_ratio * h
    x1, y1, x2, y2 = bbox
    return x1 >= margin_w and y1 >= margin_h and x2 <= (w - margin_w) and y2 <= (h - margin_h)


def _is_high_motion(imu_row: dict[str, float] | None, threshold: float) -> bool:
    if imu_row is None:
        return False
    gx = float(imu_row.get("gx", 0.0))
    gy = float(imu_row.get("gy", 0.0))
    gz = float(imu_row.get("gz", 0.0))
    return float(np.sqrt(gx * gx + gy * gy + gz * gz)) >= threshold


def _color_for_label(label: str) -> tuple[int, int, int]:
    seed = abs(hash(label)) % 255
    return (50 + (seed * 29) % 205, 50 + (seed * 53) % 205, 50 + (seed * 97) % 205)


def _track_shape_center_compatible(
    prev_bbox: tuple[float, float, float, float],
    new_bbox: tuple[float, float, float, float],
    config: Phase3Config,
) -> bool:
    px1, py1, px2, py2 = prev_bbox
    nx1, ny1, nx2, ny2 = new_bbox
    pw, ph = max(1.0, px2 - px1), max(1.0, py2 - py1)
    nw, nh = max(1.0, nx2 - nx1), max(1.0, ny2 - ny1)
    p_area = pw * ph
    n_area = nw * nh
    size_ratio = n_area / p_area
    min_r, max_r = config.track_size_ratio_range
    if size_ratio < min_r or size_ratio > max_r:
        return False
    pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
    ncx, ncy = (nx1 + nx2) * 0.5, (ny1 + ny2) * 0.5
    dist = float(np.hypot(ncx - pcx, ncy - pcy))
    diag = float(np.hypot(pw, ph))
    return dist <= config.track_center_dist_ratio * diag


def _passes_scene_priors(
    class_name: str,
    bbox_xyxy: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
    config: Phase3Config,
) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area_ratio = (bw * bh) / float(frame_w * frame_h)
    aspect = bw / bh
    if class_name in config.class_area_range_ratio:
        lo, hi = config.class_area_range_ratio[class_name]
        if area_ratio < lo or area_ratio > hi:
            return False
    if class_name in config.class_aspect_range:
        lo, hi = config.class_aspect_range[class_name]
        if aspect < lo or aspect > hi:
            return False
    return True
