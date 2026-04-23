from __future__ import annotations

import pytest
import pandas as pd

from trekion.sync import build_frame_sync


def test_build_frame_sync_picks_nearest_imu_sample() -> None:
    imu = pd.DataFrame(
        {
            "timestamp": [0.0, 10.0, 20.0, 30.0],
            "ax": [0, 0, 0, 0],
            "ay": [0, 0, 0, 0],
            "az": [0, 0, 0, 0],
            "gx": [0, 0, 0, 0],
            "gy": [0, 0, 0, 0],
            "gz": [0, 0, 0, 0],
            "mx": [0, 0, 0, 0],
            "my": [0, 0, 0, 0],
            "mz": [0, 0, 0, 0],
            "temp": [0, 0, 0, 0],
        }
    )
    vts = pd.DataFrame({"frame_idx": [0, 1, 2], "timestamp": [2.0, 18.0, 28.0]})
    synced = build_frame_sync(vts, imu)
    assert list(synced["imu_idx"]) == [0, 2, 3]
    assert "signed_delay" in synced.columns
    assert "is_boundary_clamped" in synced.columns


def test_build_frame_sync_rejects_too_short_imu() -> None:
    imu = pd.DataFrame(
        {
            "timestamp": [0.0],
            "ax": [0.0],
            "ay": [0.0],
            "az": [0.0],
            "gx": [0.0],
            "gy": [0.0],
            "gz": [0.0],
            "mx": [0.0],
            "my": [0.0],
            "mz": [0.0],
            "temp": [0.0],
        }
    )
    vts = pd.DataFrame({"frame_idx": [0], "timestamp": [0.0]})
    with pytest.raises(ValueError):
        build_frame_sync(vts, imu)

