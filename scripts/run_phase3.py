from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trekion.segmentation import (
    Phase3Config,
    fuse_scene_and_hands,
    infer_detections,
    infer_hands,
    render_detections,
    segmentation_pipeline,
)
from trekion.parsers import parse_imu, parse_vts
from trekion.sync import build_frame_sync

DEFAULT_YOLO = "yolov8s-seg.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3: instance segmentation + optional hand landmarks.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_YOLO)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--hands", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--imu", type=Path, default=None)
    p.add_argument("--vts", type=Path, default=None)
    p.add_argument("--motion-threshold", type=float, default=4.0)
    p.add_argument("--max-frames", type=int, default=-1)
    return p.parse_args()


def _load_imu_sync(imu_path: Path | None, vts_path: Path | None) -> pd.DataFrame | None:
    if imu_path is None or vts_path is None:
        return None
    imu_df = parse_imu(imu_path).dataframe
    vts_df = parse_vts(vts_path)
    synced = build_frame_sync(vts_df, imu_df)
    idx = synced["imu_idx"].to_numpy(dtype=int)
    synced["gx"] = imu_df.iloc[idx]["gx"].to_numpy()
    synced["gy"] = imu_df.iloc[idx]["gy"].to_numpy()
    synced["gz"] = imu_df.iloc[idx]["gz"].to_numpy()
    return synced


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = Phase3Config(
        model_id=args.model,
        device=args.device,
        conf=args.conf,
        imgsz=args.imgsz,
        enable_hands=args.hands,
        enable_depth=args.depth,
        motion_gyro_threshold=args.motion_threshold,
        hand_model_cache_dir=ROOT / "models",
    )
    pipeline = segmentation_pipeline(config)
    frame_sync = _load_imu_sync(args.imu, args.vts)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output: {args.output}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))

            detections, infer_ms, stage_counts = infer_detections(pipeline, frame)
            hands = infer_hands(pipeline, frame, timestamp_ms)

            imu_row = None
            if frame_sync is not None and frame_idx < len(frame_sync):
                row = frame_sync.iloc[frame_idx]
                imu_row = {"gx": float(row.get("gx", 0.0)), "gy": float(row.get("gy", 0.0)), "gz": float(row.get("gz", 0.0))}

            fused = fuse_scene_and_hands(
                pipeline=pipeline,
                detections=detections,
                hands=hands,
                frame_bgr=frame,
                frame_index=frame_idx,
                timestamp_ms=timestamp_ms,
                imu_row=imu_row,
                stage_counts=stage_counts,
                inference_ms=infer_ms,
            )
            annotated = render_detections(frame, fused)

            writer.write(annotated)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"processed {frame_idx} frames")
    finally:
        cap.release()
        writer.release()
        if pipeline.hand_landmarker is not None:
            pipeline.hand_landmarker.close()

    print(f"saved output -> {args.output}")


if __name__ == "__main__":
    main()
