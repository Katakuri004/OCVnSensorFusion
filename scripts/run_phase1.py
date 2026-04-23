from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trekion.parsers import parse_imu, parse_vts
from trekion.sync import (
    build_frame_sync,
    compute_sampling_rate,
    compute_timing_diagnostics,
    get_imu_window,
)
from trekion.visualization import draw_hud, render_imu_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trekion Phase 1 pipeline.")
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--vts", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-window", type=float, default=2.0, help="Window half-size.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Debug limiter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    imu_result = parse_imu(args.imu)
    imu = imu_result.dataframe
    vts = parse_vts(args.vts)
    sync = build_frame_sync(vts, imu)
    print(f"IMU parser diagnostics: {imu.attrs.get('parser_diagnostics', {})}")
    print(f"VTS parser diagnostics: {vts.attrs.get('parser_diagnostics', {})}")

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
        raise RuntimeError(f"Failed to open output writer: {args.output}")

    imu_hz = compute_sampling_rate(imu["timestamp"].to_numpy())
    unclamped = sync.loc[~sync["is_boundary_clamped"]]
    sync_source = unclamped if len(unclamped) > 0 else sync
    sync_stats = {
        "mean": float(sync_source["sync_delay"].mean() * 1000.0),
        "median": float(sync_source["sync_delay"].median() * 1000.0),
        "max": float(sync_source["sync_delay"].max() * 1000.0),
    }
    imu_diag = compute_timing_diagnostics(imu["timestamp"].to_numpy())
    frame_diag = compute_timing_diagnostics(vts["timestamp"].to_numpy())
    boundary_clamped_pct = float(sync["is_boundary_clamped"].mean() * 100.0)
    signed_delay_mean_ms = float(sync["signed_delay"].mean() * 1000.0)

    print(
        "IMU timing(ms): "
        f"mean={imu_diag['dt_mean_ms']:.3f} std={imu_diag['dt_std_ms']:.3f} "
        f"p95={imu_diag['dt_p95_ms']:.3f} max={imu_diag['dt_max_ms']:.3f} "
        f"non_increasing={int(imu_diag['non_increasing_count'])}"
    )
    print(
        "VTS timing(ms): "
        f"mean={frame_diag['dt_mean_ms']:.3f} std={frame_diag['dt_std_ms']:.3f} "
        f"p95={frame_diag['dt_p95_ms']:.3f} max={frame_diag['dt_max_ms']:.3f} "
        f"non_increasing={int(frame_diag['non_increasing_count'])}"
    )
    print(
        "Sync summary: "
        f"mean_abs_ms={sync_stats['mean']:.3f} median_abs_ms={sync_stats['median']:.3f} "
        f"max_abs_ms={sync_stats['max']:.3f} signed_mean_ms={signed_delay_mean_ms:.3f} "
        f"boundary_clamped_pct={boundary_clamped_pct:.3f}"
    )
    outliers = sync.nlargest(10, "sync_delay")[
        ["frame_idx", "timestamp", "imu_idx", "signed_delay", "sync_delay"]
    ]
    print("Top 10 sync outliers (delay_ms):")
    for row in outliers.itertuples(index=False):
        print(
            f"  frame={int(row.frame_idx)} frame_ts={row.timestamp:.6f}s "
            f"imu_idx={int(row.imu_idx)} signed_ms={row.signed_delay*1000.0:.3f} "
            f"abs_ms={row.sync_delay*1000.0:.3f}"
        )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx >= len(sync):
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        frame_sync = sync.iloc[frame_idx]
        imu_row = imu.iloc[int(frame_sync["imu_idx"])]
        frame_ts = float(frame_sync["timestamp"])
        imu_window = get_imu_window(imu, frame_ts, args.plot_window)

        hud_frame = draw_hud(
            frame=frame,
            frame_idx=frame_idx,
            frame_ts=frame_ts,
            imu_row=imu_row,
            imu_hz=imu_hz,
            camera_fps=fps,
            sync_stats=sync_stats,
        )
        panel = render_imu_panel(
            window=imu_window,
            width=width,
            height=height,
            center_ts=frame_ts,
        )
        composed = np.hstack([hud_frame, panel])
        writer.write(composed)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"processed {frame_idx} frames")

    cap.release()
    writer.release()
    print(f"saved output -> {args.output}")


if __name__ == "__main__":
    main()

