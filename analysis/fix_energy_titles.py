import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HW = "/Users/coalminer/Desktop/流体模拟/HW1"
SRC = os.path.join(HW, "energy_png")
DST = os.path.join(HW, "energy_png_fixed")
os.makedirs(DST, exist_ok=True)

FILES = [
    ("E_a0.05_ap0.05_k1.png", "perturbation kinetic energy,  a = 0.05"),
    ("E_a0.08_ap0.08_k1.png", "perturbation kinetic energy,  a = 0.08"),
    ("E_a0.1_ap0.1_k1.png", "perturbation kinetic energy,  a = 0.1"),
    ("E_a0.2_ap0.2_k1.png", "perturbation kinetic energy,  a = 0.2"),
]


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


for name, title in FILES:
    path = os.path.join(SRC, name)
    im = Image.open(path).convert("RGB")
    gray = np.asarray(im.convert("L"))
    dark_per_row = (gray < 200).sum(axis=1)
    width = gray.shape[1]
    # top spine of the axes: first row whose dark run spans most of the width
    candidates = np.flatnonzero(dark_per_row > 0.6 * width)
    if candidates.size == 0:
        raise SystemExit(f"{name}: could not locate the axes top spine")
    spine = int(candidates[0])
    im.paste((255, 255, 255), (0, 0, width, spine))

    d = ImageDraw.Draw(im)
    size = max(24, min(64, spine - 14))
    f = font(size)
    tw = d.textlength(title, font=f)
    d.text(((width - tw) / 2, (spine - size * 1.15) / 2), title, font=f, fill=(0, 0, 0))
    out = os.path.join(DST, name)
    im.save(out)
    print(f"{name}: spine_row={spine} title_size={size} -> {out}")
