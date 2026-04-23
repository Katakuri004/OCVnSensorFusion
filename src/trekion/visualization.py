from __future__ import annotations

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def render_imu_panel(window: pd.DataFrame, width: int, height: int, center_ts: float) -> np.ndarray:
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    fig.patch.set_facecolor("#121212")
    canvas = FigureCanvas(fig)
    axs = fig.subplots(3, 1, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.08, hspace=0.28)

    if window.empty:
        labels = [("ax", "ay", "az"), ("gx", "gy", "gz"), ("mx", "my", "mz")]
        for ax, names in zip(axs, labels):
            for n in names:
                ax.plot([], [], label=n)
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.2)
    else:
        t = window["timestamp"].to_numpy() - center_ts
        triplets = [("ax", "ay", "az"), ("gx", "gy", "gz"), ("mx", "my", "mz")]
        titles = ["Acceleration", "Gyroscope", "Magnetometer"]
        for ax, names, title in zip(axs, triplets, titles):
            ax.set_facecolor("#1b1b1b")
            for n in names:
                ax.plot(t, window[n].to_numpy(), label=n, linewidth=1.5)
            ax.axvline(0.0, color="white", linewidth=1.0, alpha=0.5)
            ax.set_title(title, fontsize=10, color="white")
            ax.legend(loc="upper left", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#888888")
        axs[-1].set_xlabel("time offset (s)", color="white", fontsize=9)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    panel = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return panel


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    frame_ts: float,
    imu_row: pd.Series,
    imu_hz: float,
    camera_fps: float,
    sync_stats: dict[str, float],
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (1080, 182), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0.0)

    lines = [
        f"frame={frame_idx} ts={frame_ts:.6f}s",
        (
            "acc=({:+.3f},{:+.3f},{:+.3f}) gyro=({:+.3f},{:+.3f},{:+.3f})".format(
                imu_row["ax"],
                imu_row["ay"],
                imu_row["az"],
                imu_row["gx"],
                imu_row["gy"],
                imu_row["gz"],
            )
        ),
        "mag=({:+.3f},{:+.3f},{:+.3f}) temp={:.2f}C".format(
            imu_row["mx"], imu_row["my"], imu_row["mz"], imu_row["temp"]
        ),
        f"imu_hz={imu_hz:.2f} cam_fps={camera_fps:.2f}",
        "sync_delay_ms mean={:.3f} median={:.3f} max={:.3f}".format(
            sync_stats["mean"], sync_stats["median"], sync_stats["max"]
        ),
    ]

    y = 34
    for line in lines:
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        y += 30
    return out

