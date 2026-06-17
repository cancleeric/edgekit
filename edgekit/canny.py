"""The Canny edge detector, stage by stage, in NumPy.

Canny (1986) is four steps after a smoothing blur:
  1. gradient magnitude + direction (Sobel)
  2. non-maximum suppression -> thin the ridges to 1px
  3. double threshold -> mark strong vs weak edges
  4. hysteresis -> keep weak edges only if connected to a strong one

Each step is a separate function so the pipeline is inspectable; `canny()`
returns the final edge map and a dict of every intermediate stage for
visualisation.
"""

from __future__ import annotations

import numpy as np

from .filters import gaussian_blur, sobel_gradients


def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Thin edges: keep a pixel only if it is a local max along the gradient."""
    h, w = magnitude.shape
    out = np.zeros_like(magnitude)
    angle = np.rad2deg(direction) % 180  # gradient orientation, 0..180

    padded = np.pad(magnitude, 1, mode="constant")
    for i in range(h):
        for j in range(w):
            a = angle[i, j]
            m = magnitude[i, j]
            # pick the two neighbours straddling the gradient direction
            if a < 22.5 or a >= 157.5:        # horizontal gradient
                n1, n2 = padded[i + 1, j], padded[i + 1, j + 2]
            elif a < 67.5:                    # diagonal /
                n1, n2 = padded[i, j + 2], padded[i + 2, j]
            elif a < 112.5:                   # vertical gradient
                n1, n2 = padded[i, j + 1], padded[i + 2, j + 1]
            else:                             # diagonal \
                n1, n2 = padded[i, j], padded[i + 2, j + 2]
            if m >= n1 and m >= n2:
                out[i, j] = m
    return out


def double_threshold(image: np.ndarray, low: float, high: float):
    """Split pixels into strong (>=high) and weak (low..high) edges."""
    strong = image >= high
    weak = (image >= low) & (image < high)
    return strong, weak


def hysteresis(strong: np.ndarray, weak: np.ndarray) -> np.ndarray:
    """Promote weak pixels that touch (8-connectivity) a strong/edge pixel.

    Iterates to a fixed point so chains of weak pixels anchored to a strong one
    are all kept, while isolated weak pixels are dropped.
    """
    edges = strong.copy()
    while True:
        # any weak pixel adjacent to a current edge becomes an edge
        neighbour = np.zeros_like(edges)
        neighbour[1:, :] |= edges[:-1, :]
        neighbour[:-1, :] |= edges[1:, :]
        neighbour[:, 1:] |= edges[:, :-1]
        neighbour[:, :-1] |= edges[:, 1:]
        neighbour[1:, 1:] |= edges[:-1, :-1]
        neighbour[:-1, :-1] |= edges[1:, 1:]
        neighbour[1:, :-1] |= edges[:-1, 1:]
        neighbour[:-1, 1:] |= edges[1:, :-1]

        promote = weak & neighbour & ~edges
        if not promote.any():
            break
        edges |= promote
    return edges


def canny(image: np.ndarray, sigma: float = 1.4, low: float = 0.08,
          high: float = 0.20, kernel_size: int = 5):
    """Run the full pipeline. Returns (edges_bool, stages_dict).

    Thresholds are fractions of the suppressed gradient's max, so they are
    scale-independent of the input image's contrast.
    """
    blurred = gaussian_blur(image, kernel_size, sigma)
    _, _, magnitude, direction = sobel_gradients(blurred)
    suppressed = non_max_suppression(magnitude, direction)

    norm = suppressed / (suppressed.max() + 1e-12)
    strong, weak = double_threshold(norm, low, high)
    edges = hysteresis(strong, weak)

    stages = {
        "blurred": blurred,
        "magnitude": magnitude,
        "suppressed": suppressed,
        "edges": edges.astype(float),
    }
    return edges, stages
