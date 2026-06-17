"""Unit tests for the filters and the Canny pipeline (synthetic inputs)."""

import numpy as np

from edgekit import (
    canny,
    correlate2d,
    gaussian_kernel,
    non_max_suppression,
    sobel_gradients,
)


def test_gaussian_kernel_normalised():
    k = gaussian_kernel(5, 1.0)
    assert k.shape == (5, 5)
    assert np.isclose(k.sum(), 1.0)
    # peak at the centre
    assert k[2, 2] == k.max()


def test_correlate_identity():
    img = np.random.rand(10, 12)
    identity = np.zeros((3, 3)); identity[1, 1] = 1.0
    assert np.allclose(correlate2d(img, identity), img)


def test_sobel_finds_vertical_edge():
    # left half dark, right half bright -> strong horizontal gradient at seam
    img = np.zeros((20, 20)); img[:, 10:] = 1.0
    gx, gy, mag, _ = sobel_gradients(img)
    assert mag[:, 10].mean() > mag[:, 2].mean()      # response at the edge
    assert abs(gx).sum() > abs(gy).sum()             # gradient is horizontal


def test_nms_thins_a_thick_band():
    # a ridge that peaks in the middle column (not a flat plateau)
    mag = np.zeros((20, 20))
    mag[:, 8:13] = np.array([0.3, 0.6, 1.0, 0.6, 0.3])  # peak at column 10
    direction = np.zeros_like(mag)                       # horizontal gradient
    thin = non_max_suppression(mag, direction)
    assert (thin > 0).sum() < (mag > 0).sum()            # ridge got thinner
    assert (thin[:, 10] > 0).all()                       # the peak survives


def test_canny_on_square_returns_outline():
    img = np.zeros((40, 40)); img[10:30, 10:30] = 1.0
    edges, stages = canny(img, sigma=1.0, low=0.1, high=0.3)
    assert edges.dtype == bool
    assert 0 < edges.sum() < img.size              # some, but not all, pixels
    assert set(stages) == {"blurred", "magnitude", "suppressed", "edges"}
    # edges should trace the border, not fill the interior
    assert not edges[19, 19]
