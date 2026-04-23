from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sampling_rate(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 0.0
    diffs = np.diff(timestamps.astype(np.float64))
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return 0.0
    return 1.0 / median_dt


def compute_timing_diagnostics(timestamps: np.ndarray) -> dict[str, float]:
    if len(timestamps) < 2:
        return {
            "count": float(len(timestamps)),
            "dt_mean_ms": 0.0,
            "dt_std_ms": 0.0,
            "dt_min_ms": 0.0,
            "dt_p95_ms": 0.0,
            "dt_max_ms": 0.0,
            "non_increasing_count": 0.0,
        }

    diffs = np.diff(timestamps.astype(np.float64))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return {
            "count": float(len(timestamps)),
            "dt_mean_ms": 0.0,
            "dt_std_ms": 0.0,
            "dt_min_ms": 0.0,
            "dt_p95_ms": 0.0,
            "dt_max_ms": 0.0,
            "non_increasing_count": float(np.sum(diffs <= 0)),
        }

    return {
        "count": float(len(timestamps)),
        "dt_mean_ms": float(np.mean(positive) * 1000.0),
        "dt_std_ms": float(np.std(positive) * 1000.0),
        "dt_min_ms": float(np.min(positive) * 1000.0),
        "dt_p95_ms": float(np.percentile(positive, 95) * 1000.0),
        "dt_max_ms": float(np.max(positive) * 1000.0),
        "non_increasing_count": float(np.sum(diffs <= 0)),
    }


def build_frame_sync(vts: pd.DataFrame, imu: pd.DataFrame) -> pd.DataFrame:
    imu_ts = imu["timestamp"].to_numpy(dtype=np.float64)
    frame_ts = vts["timestamp"].to_numpy(dtype=np.float64)
    if len(imu_ts) < 2:
        raise ValueError("IMU stream must contain at least 2 timestamps for synchronization.")
    if len(frame_ts) == 0:
        raise ValueError("VTS stream must contain at least 1 timestamp for synchronization.")

    indices = np.searchsorted(imu_ts, frame_ts, side="left")
    left_clamped = indices == 0
    right_clamped = indices == len(imu_ts)
    indices = np.clip(indices, 1, len(imu_ts) - 1)
    left = indices - 1
    right = indices
    choose_right = (imu_ts[right] - frame_ts) < (frame_ts - imu_ts[left])
    nearest = np.where(choose_right, right, left)

    signed_delay = imu_ts[nearest] - frame_ts
    sync_delay = np.abs(signed_delay)
    out = vts.copy()
    out["imu_idx"] = nearest
    out["signed_delay"] = signed_delay
    out["sync_delay"] = sync_delay
    out["left_clamped"] = left_clamped
    out["right_clamped"] = right_clamped
    out["is_boundary_clamped"] = left_clamped | right_clamped
    return out


def get_imu_window(imu: pd.DataFrame, center_ts: float, half_window: float) -> pd.DataFrame:
    start = center_ts - half_window
    end = center_ts + half_window
    ts = imu["timestamp"].to_numpy(dtype=np.float64)
    left = int(np.searchsorted(ts, start, side="left"))
    right = int(np.searchsorted(ts, end, side="right"))
    return imu.iloc[left:right]

