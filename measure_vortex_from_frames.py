"""
Measure vortex positions directly from perturbation frames.
Strategy: In the perturbation plots, vortices appear as closed contour regions
near y=0. We detect them by finding local maxima/minima of q' in the central band.
"""
import numpy as np
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# Domain: x in [0, 10], y in [-2, 2]
# Images are 3000x1200 pixels

# Determine plot area by examining a frame
def find_plot_area(img):
    """Find the plot area boundaries by detecting the axis frame."""
    h, w = img.shape[:2]
    # The plot area is bounded by black axis lines
    # Look for the leftmost and rightmost vertical black lines
    gray = img.mean(axis=2)
    
    # Find columns that are mostly dark (axis lines)
    col_darkness = (gray < 100).mean(axis=0)
    # The y-axis is typically around 5-10% from left
    # Find first significant dark column
    threshold = 0.3
    dark_cols = np.where(col_darkness > threshold)[0]
    
    if len(dark_cols) > 0:
        # Group consecutive dark columns
        groups = []
        current_group = [dark_cols[0]]
        for c in dark_cols[1:]:
            if c - current_group[-1] <= 3:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]
        groups.append(current_group)
        
        # First group = y-axis, last group = right edge or colorbar
        plot_left = groups[0][-1] + 2  # right edge of y-axis + small offset
        if len(groups) > 1:
            plot_right = groups[-1][0] - 2
        else:
            plot_right = w - 50  # fallback
    else:
        plot_left = int(w * 0.07)
        plot_right = int(w * 0.90)
    
    # Similarly for y-axis
    row_darkness = (gray < 100).mean(axis=1)
    dark_rows = np.where(row_darkness > threshold)[0]
    if len(dark_rows) > 0:
        groups = []
        current_group = [dark_rows[0]]
        for r in dark_rows[1:]:
            if r - current_group[-1] <= 3:
                current_group.append(r)
            else:
                groups.append(current_group)
                current_group = [r]
        groups.append(current_group)
        plot_top = groups[0][-1] + 2
        plot_bottom = groups[-1][0] - 2 if len(groups) > 1 else h - 20
    else:
        plot_top = int(h * 0.05)
        plot_bottom = int(h * 0.92)
    
    return plot_left, plot_right, plot_top, plot_bottom


def extract_q_field(img, plot_left, plot_right, plot_top, plot_bottom):
    """Extract the q' field from the colored perturbation plot.
    Red = positive q', Blue = negative q'."""
    # Crop to plot area
    crop = img[plot_top:plot_bottom, plot_left:plot_right]
    h, w = crop.shape[:2]
    
    r = crop[:, :, 0].astype(float)
    g = crop[:, :, 1].astype(float)
    b = crop[:, :, 2].astype(float)
    
    # q' ~ R - B (red positive, blue negative, white/grey zero)
    q_field = r - b
    
    return q_field


def find_vortex_x_positions(q_field, n_vortices=2, y_band_fraction=0.15):
    """Find vortex x-positions by looking at the central band of q'."""
    h, w = q_field.shape
    
    # Focus on central band (where vortices are)
    center = h // 2
    band_half = int(h * y_band_fraction)
    band = q_field[center - band_half:center + band_half, :]
    
    # Smooth
    smoothed = gaussian_filter(band, sigma=8)
    
    # Average over y to get 1D profile
    profile = smoothed.mean(axis=0)
    
    # Find peaks (positive vortex) and troughs (negative vortex)
    pos_peaks, pos_props = find_peaks(profile, distance=w//3, prominence=5)
    neg_peaks, neg_props = find_peaks(-profile, distance=w//3, prominence=5)
    
    # Combine all peaks and sort by absolute prominence
    all_peaks = []
    for p in pos_peaks:
        idx = np.where(pos_peaks == p)[0][0]
        all_peaks.append((p, pos_props['prominences'][idx]))
    for p in neg_peaks:
        idx = np.where(neg_peaks == p)[0][0]
        all_peaks.append((p, neg_props['prominences'][idx]))
    
    # Sort by prominence (descending) and take top n_vortices
    all_peaks.sort(key=lambda x: -x[1])
    top_peaks = sorted([p[0] for p in all_peaks[:n_vortices]])
    
    return top_peaks, profile


# Case definitions
cases = {
    0.05: {
        'dir': '/Users/coalminer/Downloads/frames_dt1/',
        'prefix': 'solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert_t',
        'has_t00': False,
    },
    0.08: {
        'dir': '/Users/coalminer/Downloads/frames_dt1/',
        'prefix': 'solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert_t',
        'has_t00': False,
    },
    0.10: {
        'dir': '/Users/coalminer/Downloads/frames_dt1/',
        'prefix': 'solve_2D_nonlinear_case1_ap0.1_k1_pert_t',
        'has_t00': True,
    },
    0.20: {
        'dir': '/Users/coalminer/Downloads/frames_dt1/',
        'prefix': 'solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert_t',
        'has_t00': False,
    },
}

# First, determine plot area from a reference frame
ref_img = np.array(Image.open('/Users/coalminer/Downloads/frames_dt1/solve_2D_nonlinear_case1_ap0.1_k1_pert_t10.png'))
plot_left, plot_right, plot_top, plot_bottom = find_plot_area(ref_img)
print(f"Plot area: left={plot_left}, right={plot_right}, top={plot_top}, bottom={plot_bottom}")
print(f"Plot size: {plot_right - plot_left} x {plot_bottom - plot_top}")

# Results storage
results = {}

for a_val, case_info in cases.items():
    print(f"\n=== a={a_val} ===")
    
    start_frame = 0 if case_info['has_t00'] else 1
    
    # Find max available frame
    max_frame = 500
    while max_frame > start_frame and not os.path.exists(
        os.path.join(case_info['dir'], f"{case_info['prefix']}{max_frame:02d}.png")):
        max_frame -= 1
    print(f"Frames: {start_frame} to {max_frame}")
    
    # Sample at dt=1 (every 10 frames)
    sample_frames = list(range(start_frame, max_frame + 1, 10))
    if case_info['has_t00'] and 0 not in sample_frames:
        sample_frames = [0] + sample_frames
    
    times = []
    x_positions = []  # list of [x_left, x_right]
    
    prev_positions = None
    
    for frame_num in sample_frames:
        fpath = os.path.join(case_info['dir'], f"{case_info['prefix']}{frame_num:02d}.png")
        if not os.path.exists(fpath):
            continue
        
        img = np.array(Image.open(fpath))
        
        # Extract q' field
        q_field = extract_q_field(img, plot_left, plot_right, plot_top, plot_bottom)
        
        # Find vortex positions in pixel coords
        peak_pixels, profile = find_vortex_x_positions(q_field)
        
        # Convert to domain coordinates
        plot_w = plot_right - plot_left
        domain_xs = [p / plot_w * 10.0 for p in peak_pixels]
        
        t_val = frame_num / 10.0
        
        # Track continuity: match with previous positions
        if prev_positions is not None and len(domain_xs) == 2 and len(prev_positions) == 2:
            # Match each detected vortex to the closest previous one
            matched = [None, None]
            used = [False, False]
            for i, dx in enumerate(domain_xs):
                best_j = -1
                best_dist = float('inf')
                for j, px in enumerate(prev_positions):
                    if used[j]:
                        continue
                    # Account for periodicity
                    dist = min(abs(dx - px), abs(dx - px + 10), abs(dx - px - 10))
                    if dist < best_dist:
                        best_dist = dist
                        best_j = j
                if best_j >= 0:
                    matched[best_j] = dx
                    used[best_j] = True
            
            # Fill unmatched with detected values
            for i, dx in enumerate(domain_xs):
                if dx not in matched:
                    for j in range(2):
                        if matched[j] is None:
                            matched[j] = dx
                            break
            
            domain_xs = sorted([m if m is not None else domain_xs[0] for m in matched])
        
        times.append(t_val)
        x_positions.append(domain_xs)
        prev_positions = domain_xs
        
        if len(domain_xs) >= 2:
            print(f"  t={t_val:5.1f}: x_left={domain_xs[0]:.2f}, x_right={domain_xs[1]:.2f}")
        else:
            print(f"  t={t_val:5.1f}: only {len(domain_xs)} vortex found: {domain_xs}")
    
    results[a_val] = {
        't': np.array(times),
        'x_positions': x_positions,
    }

# Save results
for a_val, res in results.items():
    xs = np.array(res['x_positions'])
    np.savez(f'vortex_tracking/results/a{a_val}_frame_vortex.npz',
             t=res['t'], x_left=xs[:, 0] if xs.shape[1] >= 1 else xs[:, 0],
             x_right=xs[:, 1] if xs.shape[1] >= 2 else xs[:, 0])
    print(f"\nSaved a={a_val}: {len(res['t'])} time points")

print("\nDone!")
