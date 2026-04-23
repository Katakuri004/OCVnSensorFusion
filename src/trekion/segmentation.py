from __future__ import annotations

from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def yolo_predict_device(flag: str) -> str | int:
    if flag == "auto":
        try:
            import torch

            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if flag == "cpu":
        return "cpu"
    return 0


def ensure_hand_landmarker_model(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "hand_landmarker.task"
    if not path.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def create_hand_landmarker(model_path: Path) -> mp.tasks.vision.HandLandmarker:
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def draw_hands_bgr(
    frame_bgr: np.ndarray,
    landmarker: mp.tasks.vision.HandLandmarker,
    timestamp_ms: int,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    out = frame_bgr.copy()
    for hand in result.hand_landmarks:
        for connection in HAND_CONNECTIONS:
            a, b = connection
            ax = int(hand[a].x * w)
            ay = int(hand[a].y * h)
            bx = int(hand[b].x * w)
            by = int(hand[b].y * h)
            cv2.line(out, (ax, ay), (bx, by), (255, 200, 0), 2)
        for lm in hand:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)
    return out
