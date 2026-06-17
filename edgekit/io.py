"""Tiny image I/O: load to a float grayscale array in [0, 1], save back."""

from __future__ import annotations

import numpy as np
from PIL import Image


def load_gray(path: str) -> np.ndarray:
    """Load any image as a float32 grayscale array scaled to [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def save_gray(array: np.ndarray, path: str) -> None:
    """Save a float [0, 1] (or boolean) array as an 8-bit grayscale PNG."""
    arr = np.asarray(array, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)
