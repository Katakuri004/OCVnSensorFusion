from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import struct

import numpy as np
import pandas as pd

IMU_MAGIC = b"TRIMU001"
VTS_MAGIC = b"TRIVTS01"
IMU_DATA_OFFSET_AFTER_MAGIC = 216
VTS_DATA_OFFSET_AFTER_MAGIC = 24
IMU_RECORD_SIZE_BYTES = 80
VTS_RECORD_SIZE_BYTES = 24

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImuParseResult:
    dataframe: pd.DataFrame
    record_size_bytes: int
    header_size_bytes: int
    offset_confidence: float


def parse_imu(path: str | Path) -> ImuParseResult:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("IMU file too small to contain magic header.")

    magic = raw[:8]
    if magic != IMU_MAGIC:
        raise ValueError(f"Unexpected IMU magic: {magic!r}")

    payload = raw[8:]
    # Reverse-engineered layout: <Q timestamp_ns><10f sensor channels> plus 32 bytes
    # of reserved/padding bytes per record (8 + 40 + 32 = 80 bytes total).
    record_size = IMU_RECORD_SIZE_BYTES
    header_offset = IMU_DATA_OFFSET_AFTER_MAGIC
    offset_confidence = 1.0
    remaining = len(payload) - header_offset
    if remaining <= 0 or remaining % record_size != 0:
        detected_offset, detected_confidence = _detect_imu_data_start(payload, record_size=record_size)
        if detected_offset != header_offset:
            raise ValueError(
                "IMU layout mismatch: expected data offset "
                f"{header_offset} after magic, heuristic detected {detected_offset}."
            )
        header_offset = detected_offset
        offset_confidence = detected_confidence
    payload = payload[header_offset:]
    n_records = len(payload) // record_size
    if n_records == 0:
        raise ValueError("No IMU records found after header.")

    timestamps = np.empty(n_records, dtype=np.float64)
    values = np.empty((n_records, 10), dtype=np.float32)
    offset = 0
    for i in range(n_records):
        ts = struct.unpack_from("<Q", payload, offset)[0]
        sensors = struct.unpack_from("<10f", payload, offset + 8)
        timestamps[i] = float(ts) / 1_000_000_000.0
        values[i] = sensors
        offset += record_size

    columns = [
        "ax",
        "ay",
        "az",
        "gx",
        "gy",
        "gz",
        "mx",
        "my",
        "mz",
        "temp",
    ]
    imu = pd.DataFrame(values, columns=columns)
    imu.insert(0, "timestamp", timestamps)

    if not np.all(np.diff(imu["timestamp"].to_numpy()) >= 0):
        raise ValueError("IMU timestamps are not monotonic non-decreasing.")

    imu.attrs["parser_diagnostics"] = {
        "data_offset_after_magic": float(header_offset),
        "offset_confidence": float(offset_confidence),
        "record_size_bytes": float(record_size),
    }
    return ImuParseResult(
        dataframe=imu,
        record_size_bytes=record_size,
        header_size_bytes=8 + header_offset,
        offset_confidence=offset_confidence,
    )


def _detect_imu_data_start(payload: bytes, record_size: int) -> tuple[int, float]:
    candidates: list[tuple[int, float]] = []
    for offset in range(0, min(256, len(payload)), 8):
        remaining = len(payload) - offset
        if remaining <= 0 or remaining % record_size != 0:
            continue
        sample_count = min(256, remaining // record_size)
        monotonic_hits = 0
        sane_temp_hits = 0
        plausible_motion_hits = 0
        low_variance_penalty = 0
        prev_ts = None
        ts_values: list[int] = []
        ax_values: list[float] = []
        for i in range(sample_count):
            rec_offset = offset + i * record_size
            ts = struct.unpack_from("<Q", payload, rec_offset)[0]
            vals = struct.unpack_from("<10f", payload, rec_offset + 8)
            temp = vals[9]
            ax = vals[0]
            gx = vals[3]
            if prev_ts is not None and ts >= prev_ts:
                monotonic_hits += 1
            prev_ts = ts
            ts_values.append(ts)
            ax_values.append(float(ax))
            if np.isfinite(temp) and -100.0 <= temp <= 200.0:
                sane_temp_hits += 1
            if np.isfinite(ax) and np.isfinite(gx) and abs(ax) < 200.0 and abs(gx) < 5000.0:
                plausible_motion_hits += 1

        if len(ts_values) > 1:
            ts_diff = np.diff(np.asarray(ts_values, dtype=np.float64))
            positive = ts_diff[ts_diff > 0]
            if len(positive) > 4:
                spread = float(np.percentile(positive, 95) / max(np.percentile(positive, 5), 1.0))
            else:
                spread = 999.0
        else:
            spread = 999.0
        if np.std(np.asarray(ax_values, dtype=np.float64)) < 1e-6:
            low_variance_penalty = 1

        monotonic_score = monotonic_hits / max(1, sample_count - 1)
        temp_score = sane_temp_hits / sample_count
        plausible_score = plausible_motion_hits / sample_count
        interval_score = 1.0 / (1.0 + max(0.0, spread - 1.0))
        confidence = (
            0.45 * monotonic_score
            + 0.20 * temp_score
            + 0.20 * plausible_score
            + 0.15 * interval_score
            - 0.10 * low_variance_penalty
        )

        if sample_count > 1 and monotonic_score >= 0.98 and temp_score > 0:
            candidates.append((offset, confidence))

    if not candidates:
        raise ValueError("Could not detect IMU record start offset.")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]


def parse_vts(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != VTS_MAGIC:
            raise ValueError(f"Unexpected VTS magic: {magic!r}")

        payload = f.read()

    header_offset = VTS_DATA_OFFSET_AFTER_MAGIC
    offset_confidence = 1.0
    remaining = len(payload) - header_offset
    if remaining <= 0 or remaining % VTS_RECORD_SIZE_BYTES != 0:
        detected_offset, detected_confidence = _detect_vts_data_start(
            payload, record_size=VTS_RECORD_SIZE_BYTES
        )
        if detected_offset != header_offset:
            raise ValueError(
                "VTS layout mismatch: expected data offset "
                f"{header_offset} after magic, heuristic detected {detected_offset}."
            )
        header_offset = detected_offset
        offset_confidence = detected_confidence
    payload = payload[header_offset:]
    records: list[tuple[int, int]] = []
    skipped_hi_nonzero = 0
    skipped_stream_zero = 0
    for offset in range(0, len(payload), VTS_RECORD_SIZE_BYTES):
        chunk = payload[offset : offset + VTS_RECORD_SIZE_BYTES]
        if len(chunk) != VTS_RECORD_SIZE_BYTES:
            break
        seq_idx, imu_like_ts, stream_id, frame_idx, frame_ts, frame_ts_hi = struct.unpack(
            "<6I", chunk
        )
        if frame_ts_hi != 0:
            skipped_hi_nonzero += 1
            continue
        if stream_id == 0:
            skipped_stream_zero += 1
            continue
        # frame_ts is in microseconds; convert to seconds in floating-point domain.
        timestamp_seconds = int(frame_ts) / 1_000_000.0
        records.append((frame_idx, timestamp_seconds))

    if not records:
        raise ValueError("No VTS records parsed after header.")

    vts = pd.DataFrame(records, columns=["frame_idx", "timestamp"])
    vts.attrs["parser_diagnostics"] = {
        "data_offset_after_magic": float(header_offset),
        "offset_confidence": float(offset_confidence),
        "record_size_bytes": float(VTS_RECORD_SIZE_BYTES),
        "timestamp_source": "frame_ts_us_to_s_from_6I_layout",
        "skipped_frame_ts_hi_nonzero": float(skipped_hi_nonzero),
        "skipped_stream_id_zero": float(skipped_stream_zero),
    }
    total_skipped = skipped_hi_nonzero + skipped_stream_zero
    if total_skipped > 0:
        LOGGER.info(
            "VTS parser skipped %d records (frame_ts_hi!=0: %d, stream_id==0: %d).",
            total_skipped,
            skipped_hi_nonzero,
            skipped_stream_zero,
        )
    if not np.all(np.diff(vts["timestamp"].to_numpy()) >= 0):
        raise ValueError("VTS timestamps are not monotonic non-decreasing.")
    return vts


def _detect_vts_data_start(payload: bytes, record_size: int) -> tuple[int, float]:
    candidates: list[tuple[int, float]] = []
    for offset in range(0, min(256, len(payload)), 4):
        remaining = len(payload) - offset
        if remaining <= 0 or remaining % record_size != 0:
            continue
        n = min(256, remaining // record_size)
        prev_frame_ts = None
        mono_frame_ts_hits = 0
        stream_hits = 0
        small_frame_hits = 0
        frame_step_hits = 0
        prev_frame_idx = None
        for i in range(n):
            rec_offset = offset + i * record_size
            seq_idx, imu_like_ts, stream_id, frame_idx, frame_ts, frame_ts_hi = struct.unpack_from(
                "<6I", payload, rec_offset
            )
            if prev_frame_ts is not None and frame_ts >= prev_frame_ts:
                mono_frame_ts_hits += 1
            prev_frame_ts = frame_ts
            if frame_idx < 100000:
                small_frame_hits += 1
            if stream_id != 0 and frame_ts_hi == 0:
                stream_hits += 1
            if prev_frame_idx is not None and frame_idx == prev_frame_idx + 1:
                frame_step_hits += 1
            prev_frame_idx = frame_idx
        mono_score = mono_frame_ts_hits / max(1, n - 1)
        size_score = small_frame_hits / n
        stream_score = stream_hits / n
        step_score = frame_step_hits / max(1, n - 1)
        confidence = 0.35 * mono_score + 0.30 * stream_score + 0.20 * step_score + 0.15 * size_score
        if (
            n > 1
            and mono_frame_ts_hits >= n - 2
            and small_frame_hits >= n - 2
            and stream_hits >= n - 2
        ):
            candidates.append((offset, confidence))

    if not candidates:
        raise ValueError("Could not detect VTS record start offset.")
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0]

