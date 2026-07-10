"""Fourier low-pass filter — shared by SerpentFlow (shared-structure extraction)."""
from __future__ import annotations

import numpy as np


def spectral_lowpass(x: np.ndarray, r_cut: int) -> np.ndarray:
    """Circular 2D Fourier low-pass filter applied to the last two dimensions.

    Zeroes out all frequency components with 2D radius > r_cut pixels, then
    reconstructs the spatial field via inverse FFT.

    Args:
        x:     Array of shape [..., H, W].
        r_cut: Cutoff radius in frequency-pixel units.

    Returns:
        Real-valued array of the same shape as x.
    """
    X  = np.fft.rfft2(x)
    H  = x.shape[-2]
    Wr = X.shape[-1]
    fy = np.fft.fftfreq(H) * H       # pixel-space frequencies, shape [H]
    fx = np.arange(Wr, dtype=float)   # rfft frequencies, shape [Wr]
    fy2d, fx2d = np.meshgrid(fy, fx, indexing="ij")
    mask = (np.sqrt(fy2d ** 2 + fx2d ** 2) <= r_cut)
    return np.fft.irfft2(X * mask, s=x.shape[-2:]).astype(np.float32)
