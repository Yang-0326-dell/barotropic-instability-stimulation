#!/usr/bin/env python3
"""Create composite figure for a=0.1, k=1 sine case."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

VIDEO_DIR = Path("/Users/coalminer/Downloads")
ENERGY_DIR = Path("energy_png")
OUTPUT_DIR = Path("composite_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_frame(video_path, t_val):
    """Extract frame at time t."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None
    frame_idx = int(round(t_val * 10))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# a=0.1, k=1 sine case (ap = amplitude perturbation)
video_sine = VIDEO_DIR / "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4"
energy_sine = ENERGY_DIR / "E_a0.1_ap0.1_k1.png"

time_steps = [0, 1, 2, 5, 10, 20, 50]
frames = []
for t in time_steps:
    print(f"Extracting t={t}...")
    frame = extract_frame(video_sine, t)
    frames.append(frame)

# Load energy image
energy_img = np.array(Image.open(energy_sine))

# Create 2x4 composite: 7 time snapshots + 1 energy = 8 panels
fig, axes = plt.subplots(4, 2, figsize=(16, 16))

positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1)]
for i, (t, frame) in enumerate(zip(time_steps, frames)):
    row, col = positions[i]
    if frame is not None:
        axes[row, col].imshow(frame)
    axes[row, col].set_title(f't={t}', fontsize=12)
    axes[row, col].axis('off')

# Last panel: energy
row, col = positions[7]
axes[row, col].imshow(energy_img)
axes[row, col].set_title('Energy', fontsize=12)
axes[row, col].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'composite_a0.1_k1_sine.png', dpi=150, bbox_inches='tight')
print("Saved composite_a0.1_k1_sine.png")
plt.close()
