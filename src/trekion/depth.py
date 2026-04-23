from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PIL import Image


def colorize_depth(
    depth: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
    lo_pct: float = 2.0,
    hi_pct: float = 98.0,
) -> np.ndarray:
    """Map a single-channel depth map to BGR uint8 using OpenCV colormap."""
    if depth.ndim != 2:
        raise ValueError("depth must be HxW")
    d = depth.astype(np.float64)
    lo = float(np.percentile(d, lo_pct))
    hi = float(np.percentile(d, hi_pct))
    if hi <= lo:
        hi = lo + 1e-6
    normed = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    gray = (normed * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


def resize_to_match(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if w == target_w and h == target_h:
        return bgr
    return cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def bgr_to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def depth_pipeline(
    model_id: str,
    device: str | int | None,
) -> Any:
    from transformers import pipeline

    if device == "auto":
        device_arg = 0 if _cuda_available() else -1
    elif device == "cpu" or device == -1:
        device_arg = -1
    elif isinstance(device, int):
        device_arg = device
    else:
        device_arg = -1

    return pipeline(
        task="depth-estimation",
        model=model_id,
        device=device_arg,
    )


def infer_depth_map(pipe: Any, frame_bgr: np.ndarray) -> np.ndarray:
    pil = bgr_to_pil_rgb(frame_bgr)
    out = pipe(pil)
    depth_img = out["depth"]
    return np.asarray(depth_img, dtype=np.float32)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
