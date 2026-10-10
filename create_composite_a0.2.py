#!/usr/bin/env python3
"""Create composite figure for a=0.2, k=1/2/3/4 cases."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

VIDEO_DIR = Path("/Users/coalminer/Downloads")
ENERGY_DIR = Path("energy_png")
OUTPUT_DIR = Path("composite_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_last_frame(video_path):
    """Extract last frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

# a=0.2, k=1/2/3/4 cases
cases = [1, 2, 3, 4]
videos = {
    1: VIDEO_DIR / "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4",
    2: VIDEO_DIR / "solve_2D_nonlinear_case1_abg0.2_ap0.2_k2_pert.mp4",
    3: VIDEO_DIR / "solve_2D_nonlinear_case1_abg0.2_ap0.2_k3_pert.mp4",
    4: VIDEO_DIR / "solve_2D_nonlinear_case1_abg0.2_ap0.2_k4_pert.mp4",
}
energy_plots = {
    1: ENERGY_DIR / "E_a0.2_ap0.2_k1.png",
    2: ENERGY_DIR / "E_a0.2_ap0.2_k2.png",
    3: ENERGY_DIR / "E_a0.2_ap0.2_k3.png",
    4: ENERGY_DIR / "E_a0.2_ap0.2_k4.png",
}

# Extract last frames
print("Extracting last frames...")
last_frames = {}
for k in cases:
    print(f"  k={k}...")
    last_frames[k] = extract_last_frame(videos[k])

# Load energy images
energy_imgs = {}
for k in cases:
    energy_imgs[k] = np.array(Image.open(energy_plots[k]))

# Create 4x2 composite: 4 cases × (last frame + energy)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))

for idx, k in enumerate(cases):
    row = idx
    # Left: last frame
    if last_frames[k] is not None:
        axes[row, 0].imshow(last_frames[k])
    axes[row, 0].set_title(f'k={k}, last frame', fontsize=12)
    axes[row, 0].axis('off')
    
    # Right: energy plot
    axes[row, 1].imshow(energy_imgs[k])
    axes[row, 1].set_title(f'k={k}, Energy', fontsize=12)
    axes[row, 1].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'composite_a0.2_k1234.png', dpi=150, bbox_inches='tight')
print("Saved composite_a0.2_k1234.png")
plt.close()
