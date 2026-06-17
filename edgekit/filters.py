"""Linear filters built from first principles with NumPy.

A 2D filter here is plain correlation (the convention OpenCV's filter2D uses):
slide a kernel over the (reflect-padded) image and take the weighted sum. The
sliding window is vectorised with stride tricks, so there are no Python loops
over pixels and it stays fast without SciPy or OpenCV.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """A normalised 2D Gaussian kernel (sums to 1)."""
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def correlate2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'same'-size 2D correlation with reflect padding."""
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    windows = sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    return correlate2d(image, gaussian_kernel(size, sigma))


# Sobel operators: derivative in one axis, smoothing in the other.
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def sobel_gradients(image: np.ndarray):
    """Return (gx, gy, magnitude, direction-in-radians)."""
    gx = correlate2d(image, SOBEL_X)
    gy = correlate2d(image, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    return gx, gy, magnitude, direction
