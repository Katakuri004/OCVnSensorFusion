from __future__ import annotations

import numpy as np

from trekion.depth import colorize_depth


def test_colorize_depth_shape_and_dtype() -> None:
    d = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    vis = colorize_depth(d)
    assert vis.shape == (3, 4, 3)
    assert vis.dtype == np.uint8
