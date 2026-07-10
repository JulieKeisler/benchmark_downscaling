"""Month covariate — shared by iDownscale, SerpentFlow, and ConsistencyModel."""
from __future__ import annotations

import numpy as np


def month_channels(months: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return [T, 2, H, W] sin/cos month channels broadcast over space.

    Args:
        months: [T] array of calendar months (1–12).
        shape:  (T, H, W) of the corresponding spatial field.
    """
    T, H, W = shape
    sin_m = np.broadcast_to(
        np.sin(2 * np.pi * months / 12)[:, None, None], (T, H, W)).copy()
    cos_m = np.broadcast_to(
        np.cos(2 * np.pi * months / 12)[:, None, None], (T, H, W)).copy()
    return np.stack([sin_m, cos_m], axis=1).astype(np.float32)  # [T, 2, H, W]
