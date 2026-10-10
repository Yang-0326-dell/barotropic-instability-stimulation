import os
from PIL import Image, ImageDraw, ImageFont

HW = "/Users/coalminer/Desktop/流体模拟/HW1"
FR = os.path.join(HW, "last_frames")
EN = os.path.join(HW, "energy_png_fixed")
OUT = os.path.join(HW, "composite")
os.makedirs(OUT, exist_ok=True)

ROWS = [
    ("a=0.05", "a0.05", "E_a0.05_ap0.05_k1.png"),
    ("a=0.08", "a0.08", "E_a0.08_ap0.08_k1.png"),
    ("a=0.1", "a0.1", "E_a0.1_ap0.1_k1.png"),
    ("a=0.2", "a0.2", "E_a0.2_ap0.2_k1.png"),
]

H = 1200
LABEL_W = 130
GAP = 24
WHITE = (255, 255, 255)


def load(path, height=H):
    im = Image.open(path).convert("RGB")
    if im.height != height:
        im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    return im


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def row_label(text):
    im = Image.new("RGB", (LABEL_W, H), WHITE)
    d = ImageDraw.Draw(im)
    f = font(56)
    w = d.textlength(text, font=f)
    im_t = Image.new("RGB", (int(w) + 8, 72), WHITE)
    dt = ImageDraw.Draw(im_t)
    dt.text((4, 4), text, font=f, fill=(0, 0, 0))
    im_t = im_t.rotate(90, expand=True, fillcolor=WHITE)
    x = (LABEL_W - im_t.width) // 2
    y = (H - im_t.height) // 2
    im.paste(im_t, (x, y))
    return im


def stitch(panels):
    w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    canvas = Image.new("RGB", (w, H), WHITE)
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + GAP
    return canvas


rows = []
for label, key, en in ROWS:
    panels = [row_label(label),
              load(os.path.join(FR, f"{key}_total.png")),
              load(os.path.join(FR, f"{key}_pert.png")),
              load(os.path.join(EN, en))]
    r = stitch(panels)
    rows.append(r)
    r.save(os.path.join(OUT, f"row_{label.replace('=', '')}.png"))
    print("row", label, r.size)

COL_TITLES = ["total q", "perturbation", "perturbation kinetic energy"]
HEADER_H = 80
HEADER_FONT = 48


def header_strip():
    im = Image.new("RGB", (W, HEADER_H), WHITE)
    d = ImageDraw.Draw(im)
    f = font(HEADER_FONT)
    # column centers (same layout as stitch): label col, then three panels separated by GAP
    x0 = LABEL_W + GAP
    centers = []
    for p in [load(os.path.join(FR, "a0.05_total.png")),
              load(os.path.join(FR, "a0.05_pert.png")),
              load(os.path.join(EN, "E_a0.05_ap0.05_k1.png"))]:
        centers.append(x0 + p.width // 2)
        x0 += p.width + GAP
    for title, cx in zip(COL_TITLES, centers):
        tw = d.textlength(title, font=f)
        d.text((cx - tw / 2, (HEADER_H - HEADER_FONT * 1.15) / 2), title, font=f, fill=(0, 0, 0))
    return im


W = max(r.width for r in rows)
tot_h = HEADER_H + GAP + sum(r.height for r in rows) + GAP * (len(rows) - 1)
grid = Image.new("RGB", (W, tot_h), WHITE)
grid.paste(header_strip(), (0, 0))
y = HEADER_H + GAP
for r in rows:
    grid.paste(r, (0, y))
    y += r.height + GAP
grid.save(os.path.join(OUT, "grid_4x3.png"))
print("grid", grid.size)

scale = 0.45
small = grid.resize((round(W * scale), round(tot_h * scale)), Image.LANCZOS)
small.save(os.path.join(OUT, "grid_4x3_preview.png"), optimize=True)
print("preview", small.size)
