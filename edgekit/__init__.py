"""edgekit: classical Canny edge detection from scratch in NumPy."""

from .canny import canny, double_threshold, hysteresis, non_max_suppression
from .filters import correlate2d, gaussian_blur, gaussian_kernel, sobel_gradients
from .io import load_gray, save_gray

__all__ = [
    "canny", "non_max_suppression", "double_threshold", "hysteresis",
    "correlate2d", "gaussian_blur", "gaussian_kernel", "sobel_gradients",
    "load_gray", "save_gray",
]
__version__ = "0.1.0"
