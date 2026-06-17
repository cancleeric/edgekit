"""Generate a synthetic test image (geometric shapes) -- no licensing issues.

    python scripts/make_sample.py   # -> sample_data/shapes.png
"""

import os

from PIL import Image, ImageDraw

os.makedirs("sample_data", exist_ok=True)
W, H = 320, 240
img = Image.new("L", (W, H), color=235)
d = ImageDraw.Draw(img)

d.rectangle([30, 40, 130, 150], fill=90)               # solid rectangle
d.ellipse([170, 30, 290, 150], outline=20, width=4)    # circle outline
d.polygon([(60, 210), (130, 210), (95, 160)], fill=40)  # triangle
d.line([(150, 200), (300, 175)], fill=10, width=3)      # straight line
d.line([(160, 70), (160, 200)], fill=10, width=2)       # vertical line

img.save("sample_data/shapes.png")
print("wrote sample_data/shapes.png", img.size)
