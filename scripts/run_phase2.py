from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trekion.depth import colorize_depth, depth_pipeline, infer_depth_map, resize_to_match

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2: monocular depth side-by-side video.")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Inference device; auto uses CUDA when available else CPU.",
    )
    p.add_argument("--max-frames", type=int, default=-1)
    return p.parse_args()


def resolve_device(s: str) -> str | int:
    if s == "auto":
        return "auto"
    if s == "cpu":
        return "cpu"
    return 0


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_size = (width * 2, height)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        out_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output: {args.output}")

    dev = resolve_device(args.device)
    pipe = depth_pipeline(args.model, dev if dev != "auto" else "auto")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        depth = infer_depth_map(pipe, frame)
        depth_vis = colorize_depth(depth, colormap=cv2.COLORMAP_INFERNO)
        depth_vis = resize_to_match(depth_vis, width, height)
        composed = np.hstack([frame, depth_vis])
        writer.write(composed)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"processed {frame_idx} frames")

    cap.release()
    writer.release()
    print(f"saved output -> {args.output}")


if __name__ == "__main__":
    main()
