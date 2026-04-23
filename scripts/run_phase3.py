from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trekion.segmentation import (
    create_hand_landmarker,
    draw_hands_bgr,
    ensure_hand_landmarker_model,
    yolo_predict_device,
)

DEFAULT_YOLO = "yolov8n-seg.pt"


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
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--hands", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-frames", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    device = yolo_predict_device(args.device)
    model = YOLO(args.model)

    landmarker = None
    if args.hands:
        model_path = ensure_hand_landmarker_model(ROOT / "models")
        landmarker = create_hand_landmarker(model_path)

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

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            annotated = results[0].plot(line_width=2, font_size=1.0)
            if landmarker is not None:
                timestamp_ms = int(round(frame_idx * 1000.0 / fps))
                annotated = draw_hands_bgr(annotated, landmarker, timestamp_ms)

            writer.write(annotated)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"processed {frame_idx} frames")
    finally:
        cap.release()
        writer.release()
        if landmarker is not None:
            landmarker.close()

    print(f"saved output -> {args.output}")


if __name__ == "__main__":
    main()
