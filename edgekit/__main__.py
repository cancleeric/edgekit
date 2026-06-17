"""CLI: run the detector on an image and write the edge map (+ optional stages).

    python -m edgekit --image sample_data/shapes.png --out edges.png
    python -m edgekit --image sample_data/shapes.png --montage montage.png
"""

from __future__ import annotations

import argparse

import numpy as np

from .canny import canny
from .io import load_gray, save_gray


def _montage(image: np.ndarray, stages: dict) -> np.ndarray:
    """Stitch input + each stage into one normalised strip for inspection."""
    def norm(a):
        a = a.astype(float)
        return (a - a.min()) / (a.max() - a.min() + 1e-12)

    panels = [norm(image), norm(stages["blurred"]), norm(stages["magnitude"]),
              norm(stages["suppressed"]), norm(stages["edges"])]
    sep = np.ones((panels[0].shape[0], 4))  # white separators
    row = [panels[0]]
    for p in panels[1:]:
        row.extend([sep, p])
    return np.hstack(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canny edge detection from scratch")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", help="write the final edge map here")
    parser.add_argument("--montage", help="write input|blur|gradient|NMS|edges strip here")
    parser.add_argument("--sigma", type=float, default=1.4)
    parser.add_argument("--low", type=float, default=0.08)
    parser.add_argument("--high", type=float, default=0.20)
    args = parser.parse_args()

    image = load_gray(args.image)
    edges, stages = canny(image, sigma=args.sigma, low=args.low, high=args.high)
    print(f"input {image.shape}  edge pixels: {int(edges.sum())} "
          f"({edges.mean()*100:.1f}% of image)")

    if args.out:
        save_gray(edges.astype(float), args.out)
        print(f"edges -> {args.out}")
    if args.montage:
        save_gray(_montage(image, stages), args.montage)
        print(f"montage -> {args.montage}")
    if not args.out and not args.montage:
        save_gray(edges.astype(float), "edges.png")
        print("edges -> edges.png")


if __name__ == "__main__":
    main()
