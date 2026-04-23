from __future__ import annotations

from pathlib import Path

import numpy as np

from trekion.segmentation import (
    FrameDiagnostics,
    FusedFrame,
    HandDetection,
    Phase3Config,
    Phase3Pipeline,
    SceneDetection,
    _apply_depth_refinement,
    _apply_nms,
    _bbox_iou,
    _cleanup_mask,
    _passes_scene_priors,
    _smooth_bbox,
    _update_tracks,
    ensure_hand_landmarker_model,
    fuse_scene_and_hands,
    infer_hands,
    render_detections,
    yolo_predict_device,
)


def test_yolo_predict_device_cpu_explicit() -> None:
    assert yolo_predict_device("cpu") == "cpu"


def test_yolo_predict_device_cuda_explicit() -> None:
    assert yolo_predict_device("cuda") == 0


def test_hand_model_download_uses_cache(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "hand_landmarker.task"
    calls: list[tuple[str, Path]] = []

    def fake_urlretrieve(url: str, dst: Path) -> None:
        calls.append((url, dst))
        dst.write_bytes(b"ok")

    monkeypatch.setattr("trekion.segmentation.urllib.request.urlretrieve", fake_urlretrieve)
    p1 = ensure_hand_landmarker_model(tmp_path)
    p2 = ensure_hand_landmarker_model(tmp_path)
    assert p1 == target
    assert p2 == target
    assert len(calls) == 1


def test_render_with_zero_hands_and_detections() -> None:
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    fused = FusedFrame(
        detections=[],
        hands=[],
        diagnostics=FrameDiagnostics(
            frame_index=1,
            timestamp_ms=10,
            inference_ms=2.0,
            active_detections=0,
            hands_count=0,
            high_motion=False,
            model_name="test-model",
            device="cpu",
        ),
    )
    out = render_detections(frame, fused)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8


def test_render_with_mock_hand_landmarks() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    hand = HandDetection(
        landmarks_norm=[(0.2, 0.2), (0.3, 0.2), (0.4, 0.2), (0.5, 0.2), (0.6, 0.2)] * 5,
        bbox_xyxy=(10, 10, 40, 40),
        confidence=0.9,
        stable=True,
    )
    fused = FusedFrame(
        detections=[],
        hands=[hand],
        diagnostics=FrameDiagnostics(0, 0, 1.0, 0, 1, False, "m", "cpu"),
    )
    out = render_detections(frame, fused)
    assert int(out.sum()) > 0


def test_class_filtering_and_remap_in_fusion() -> None:
    cfg = Phase3Config(class_persistence_frames=1, min_confirm_frames=1)
    pipe = Phase3Pipeline(config=cfg, detector=None, class_names={})
    det = SceneDetection("laptop", "display", 0.9, (10, 10, 40, 40))
    fused = fuse_scene_and_hands(
        pipeline=pipe,
        detections=[det],
        hands=[],
        frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_ms=0,
    )
    assert len(fused.detections) == 1
    assert fused.detections[0].class_name == "display"


def test_temporal_track_persistence_and_smoothing() -> None:
    cfg = Phase3Config(min_confirm_frames=1, smoothing_alpha=0.5)
    pipe = Phase3Pipeline(config=cfg, detector=None, class_names={})
    d1 = SceneDetection("laptop", "display", 0.9, (10, 10, 30, 30))
    out1 = _update_tracks([d1], pipe)
    assert len(out1) == 1
    tid = out1[0].track_id
    d2 = SceneDetection("laptop", "display", 0.9, (12, 10, 32, 30))
    out2 = _update_tracks([d2], pipe)
    assert len(out2) == 1
    assert out2[0].track_id == tid
    assert out2[0].bbox_xyxy[0] < 12


def test_postprocessing_utilities() -> None:
    dets = [
        SceneDetection("laptop", "display", 0.9, (0, 0, 20, 20)),
        SceneDetection("laptop", "display", 0.8, (1, 1, 21, 21)),
    ]
    kept = _apply_nms(dets, 0.5)
    assert len(kept) == 1
    assert _bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    sm = _smooth_bbox((0, 0, 10, 10), (10, 10, 20, 20), 0.5)
    assert sm == (5.0, 5.0, 15.0, 15.0)
    m = np.zeros((8, 8), dtype=np.uint8)
    m[2:6, 2:6] = 1
    cleaned = _cleanup_mask(m, 3, 3)
    assert cleaned.shape == m.shape


def test_depth_refinement_rejects_inconsistent_regions() -> None:
    cfg = Phase3Config(depth_spread_limit=0.1, depth_refine_classes={"display"}, depth_reject_by_spread=True)
    depth = np.zeros((20, 20), dtype=np.float32)
    depth[:, 10:] = 100.0
    det = SceneDetection("laptop", "display", 0.9, (0, 0, 20, 20))
    refined = _apply_depth_refinement([det], depth, cfg)
    assert len(refined) == 0


def test_end_to_end_smoke_on_few_frames() -> None:
    cfg = Phase3Config(min_confirm_frames=1, class_persistence_frames=1)
    pipe = Phase3Pipeline(config=cfg, detector=None, class_names={})
    for idx in range(3):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        det = SceneDetection("laptop", "display", 0.95, (8 + idx, 8, 30 + idx, 30))
        fused = fuse_scene_and_hands(
            pipeline=pipe,
            detections=[det],
            hands=[],
            frame_bgr=frame,
            frame_index=idx,
            timestamp_ms=idx * 33,
            imu_row={"gx": 0.1, "gy": 0.1, "gz": 0.1},
        )
        out = render_detections(frame, fused)
        assert out.shape == frame.shape


def test_scene_priors_reject_implausible_device_size() -> None:
    cfg = Phase3Config()
    # Device class should not take most of the frame.
    ok = _passes_scene_priors("device", (0, 0, 640, 480), 640, 480, cfg)
    assert not ok


def test_scene_priors_accept_reasonable_notebook_region() -> None:
    cfg = Phase3Config()
    ok = _passes_scene_priors("notebook", (120, 80, 360, 300), 640, 480, cfg)
    assert ok


def test_class_aware_nms_keeps_overlapping_different_classes() -> None:
    dets = [
        SceneDetection("book", "notebook", 0.9, (10, 10, 50, 50)),
        SceneDetection("cell phone", "device", 0.85, (12, 12, 48, 48)),
    ]
    kept = _apply_nms(dets, 0.5)
    assert len(kept) == 2
