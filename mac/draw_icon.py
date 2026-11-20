# SPDX-License-Identifier: Apache-2.0
# Draws the GPU pool dashboard app icon: a heatmap of GPU squares on a dark tile.
from PIL import Image, ImageDraw, ImageFilter
import sys

N = 1024
SS = 4  # supersample for smooth edges
img = Image.new("RGBA", (N * SS, N * SS), (0, 0, 0, 0))
d = ImageDraw.Draw(img)


def s(v: float) -> int:
    """Scale a 1024-grid coordinate to the supersampled canvas."""
    return int(v * SS)


# macOS icon grid: 824 px body centered on a 1024 canvas, ~185 px corner radius.
x0, y0, x1, y1 = 100, 100, 924, 924
shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
ImageDraw.Draw(shadow).rounded_rectangle(
    [s(x0), s(y0 + 14), s(x1), s(y1 + 14)], s(185), fill=(0, 0, 0, 110)
)
shadow = shadow.filter(ImageFilter.GaussianBlur(s(18)))
img.alpha_composite(shadow)

# Vertical gradient body.
body = Image.new("RGBA", img.size, (0, 0, 0, 0))
bd = ImageDraw.Draw(body)
top, bottom = (28, 36, 52), (12, 16, 26)
for y in range(s(y0), s(y1)):
    t = (y - s(y0)) / (s(y1) - s(y0))
    bd.line(
        [(s(x0), y), (s(x1), y)],
        fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom, strict=True))
        + (255,),
    )
mask = Image.new("L", img.size, 0)
ImageDraw.Draw(mask).rounded_rectangle([s(x0), s(y0), s(x1), s(y1)], s(185), fill=255)
img.paste(body, (0, 0), mask)

# Heatmap: 4 rows (hosts) x 4 columns (GPUs), shaded by load like the dashboard.
track, heat = (44, 54, 72), (110, 163, 255)
loads = [
    [0.95, 0.80, 0.30, 0.75],
    [0.15, 0.60, 1.00, 0.45],
    [0.70, 0.05, 0.55, 0.90],
    [0.35, 0.85, 0.20, 0.65],
]
gx0, gy0, cell, gap = 215, 215, 132, 22
for r, row in enumerate(loads):
    for c, p in enumerate(row):
        color = tuple(int(a + (b - a) * p) for a, b in zip(track, heat, strict=True))
        cx, cy = gx0 + c * (cell + gap), gy0 + r * (cell + gap)
        d.rounded_rectangle(
            [s(cx), s(cy), s(cx + cell), s(cy + cell)], s(24), fill=color + (255,)
        )
        # Memory line along the bottom of each square.
        mem = (p * 0.6 + 0.35) % 1.0
        d.rounded_rectangle(
            [
                s(cx + 14),
                s(cy + cell - 22),
                s(cx + 14 + (cell - 28) * mem),
                s(cy + cell - 12),
            ],
            s(5),
            fill=(230, 236, 232, 170),
        )

# Live indicator.
d.ellipse([s(800), s(158), s(838), s(196)], fill=(76, 194, 127, 255))

out = img.resize((N, N), Image.LANCZOS)
out.save(sys.argv[1])
