from __future__ import annotations

from trekion.segmentation import yolo_predict_device


def test_yolo_predict_device_cpu_explicit() -> None:
    assert yolo_predict_device("cpu") == "cpu"


def test_yolo_predict_device_cuda_explicit() -> None:
    assert yolo_predict_device("cuda") == 0
