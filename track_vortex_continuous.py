"""
Track positive vortex (red) with narrow search window.
Key insight: vortex moves ~0.1-0.6 Lx per time unit, so search window should be < 1 Lx.
"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

Lx = 10.0
T_total = 50.0
N_frames = 501
dt_frame = T_total / (N_frames - 1)

X_PIX_MIN, X_PIX_MAX = 100, 2800
Y_PIX_MIN, Y_PIX_MAX = 100, 1100
PIX_PER_LX = (X_PIX_MAX - X_PIX_MIN) / Lx

def pix_to_x(px):
    return (px - X_PIX_MIN) / (X_PIX_MAX - X_PIX_MIN) * Lx

DL = os.path.expanduser("~/Downloads")
VIDEOS = {
    0.05: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert.mp4"),
    0.08: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert.mp4"),
    0.10: os.path.join(DL, "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4"),
    0.20: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4"),
}

def get_red_profile(frame_bgr, band_half=25):
    """Get column-wise red (positive q') strength near y=0."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    h, w = red_mask.shape
    crop = np.zeros_like(red_mask)
    crop[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX] = red_mask[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]

    y_center = (Y_PIX_MIN + Y_PIX_MAX) // 2
    band = crop[y_center - band_half:y_center + band_half, :]
    profile = band.sum(axis=0).astype(float)

    kernel = np.ones(9) / 9
    profile = np.convolve(profile, kernel, mode='same')
    return profile


def track_red_centroid(profile, prev_x_pix, search_lx=0.8):
    """Track red centroid with narrow search window."""
    if prev_x_pix is None:
        return None

    search_pix = int(search_lx * PIX_PER_LX)
    domain_width = X_PIX_MAX - X_PIX_MIN

    # Extend profile for periodic wrap
    profile_ext = np.concatenate([profile, profile[:search_pix + 50]])

    search_left = int(prev_x_pix - search_pix)
    if search_left < X_PIX_MIN:
        search_left += domain_width
    search_right = search_left + 2 * search_pix

    if search_right > len(profile_ext):
        search_right = len(profile_ext)

    region = profile_ext[search_left:search_right]
    if len(region) == 0 or region.sum() < 20:
        return None

    # Centroid
    x_local = (np.arange(len(region)) * region).sum() / region.sum()
    x_pix = search_left + x_local

    # Wrap back
    while x_pix >= X_PIX_MAX:
        x_pix -= domain_width
    while x_pix < X_PIX_MIN:
        x_pix += domain_width

    return x_pix


results = {}

for a_val, vpath in VIDEOS.items():
    print(f"\n=== a={a_val} ===")
    if not os.path.exists(vpath):
        print(f"  Video not found: {vpath}")
        continue

    cap = cv2.VideoCapture(vpath)

    # Find initial position
    # For a=0.20, perturbation appears late (t~15-20), so scan all frames
    # Use wider band for initial detection since vortex may be broad
    print("  Scanning for initial vortex...")
    prev_x_pix = None
    init_frame = 0
    y_center = (Y_PIX_MIN + Y_PIX_MAX) // 2

    for frame_idx in range(N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        # Use wide band for initial detection
        band_wide = red_mask[y_center-100:y_center+100, X_PIX_MIN:X_PIX_MAX]
        mid = (X_PIX_MIN + X_PIX_MAX) // 2
        region = band_wide[:, mid:]
        if region.sum() > 500:
            x_local = (np.arange(len(region.flatten())) * region.flatten()).sum() / region.sum()
            # Convert from 2D region to 1D x position
            col_sum = region.sum(axis=0)
            if col_sum.sum() > 500:
                x_local = (np.arange(len(col_sum)) * col_sum).sum() / col_sum.sum()
                prev_x_pix = mid + x_local
                init_frame = frame_idx
                t_init = frame_idx * dt_frame
                print(f"  Found at frame {frame_idx} (t={t_init:.1f}), x={pix_to_x(prev_x_pix):.2f}")
                break

    if prev_x_pix is None:
        print("  Could not find initial vortex")
        cap.release()
        continue

    # Track forward
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    band_half = max(25, int(a_val * 500))

    all_x_pix = [prev_x_pix]
    all_t = [init_frame * dt_frame]

    for i in range(init_frame + 1, N_frames):
        ok, frame = cap.read()
        if not ok:
            break

        t = i * dt_frame
        profile = get_red_profile(frame, band_half=band_half)
        x_pix = track_red_centroid(profile, prev_x_pix, search_lx=0.8)

        if x_pix is not None:
            all_x_pix.append(x_pix)
            all_t.append(t)
            prev_x_pix = x_pix

    cap.release()

    # Convert to domain coordinates
    x_domain = np.array([pix_to_x(px) for px in all_x_pix])
    t_arr = np.array(all_t)

    # Subsample to dt=1
    t_sub = []
    x_sub = []
    for ti, xi in zip(t_arr, x_domain):
        t_int = round(ti)
        if not t_sub or abs(t_int - t_sub[-1]) > 0.5:
            t_sub.append(t_int)
            x_sub.append(xi)

    results[a_val] = {'t': np.array(t_sub), 'x': np.array(x_sub)}
    print(f"  {len(t_sub)} samples at dt=1")
    for ti, xi in zip(t_sub, x_sub):
        print(f"  t={ti:3d}: x={xi:.3f}")

# Save
for a_val, res in results.items():
    np.savez(f'vortex_tracking/results/a{a_val}_tracked.npz', t=res['t'], x=res['x'])

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
colors = {0.05: 'blue', 0.08: 'green', 0.10: 'orange', 0.20: 'red'}
labels_map = {0.05: 'a=0.05', 0.08: 'a=0.08', 0.10: 'a=0.10', 0.20: 'a=0.20'}

for a_val, res in results.items():
    t = res['t']
    x = res['x']
    x_unwrap = np.unwrap(x, period=Lx)
    v = np.diff(x_unwrap) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2

    axes[0].plot(t, x, 'o-', color=colors[a_val], label=labels_map[a_val], markersize=4, linewidth=1.5)
    axes[1].plot(t, x_unwrap, 'o-', color=colors[a_val], label=labels_map[a_val], markersize=4, linewidth=1.5)
    axes[2].plot(t_mid, v, 'o-', color=colors[a_val], label=labels_map[a_val], markersize=4, linewidth=1.5)

axes[0].set_title('Vortex Position (Periodic Domain)', fontsize=14)
axes[0].set_ylabel('x-position (Lx)', fontsize=12)
axes[0].set_xlim(0, 50)
axes[0].set_ylim(-0.5, 10.5)
axes[0].grid(True, alpha=0.3)
axes[0].legend()
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=10, color='k', linestyle='--', alpha=0.3)

axes[1].set_title('Unwrapped Vortex Position', fontsize=14)
axes[1].set_ylabel('Unwrapped x (Lx)', fontsize=12)
axes[1].set_xlabel('Time (t)', fontsize=12)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].set_title('Vortex Velocity (Unwrapped)', fontsize=14)
axes[2].set_ylabel('Velocity (Lx/t)', fontsize=12)
axes[2].set_xlabel('Time (t)', fontsize=12)
axes[2].grid(True, alpha=0.3)
axes[2].legend()
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('vortex_tracking/results/vortex_tracking_continuous.png', dpi=150, bbox_inches='tight')
print("\nSaved vortex_tracking_continuous.png")
plt.close()

print("\n=== Velocity Summary ===")
for a_val, res in results.items():
    t = res['t']
    x = res['x']
    x_unwrap = np.unwrap(x, period=Lx)
    v = np.diff(x_unwrap) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2

    mask = t_mid > 10
    if mask.sum() > 0:
        v_stable = v[mask]
        print(f"a={a_val}: mean v = {v_stable.mean():.4f}, std = {v_stable.std():.4f}")

    coeffs = np.polyfit(t, x_unwrap, 1)
    period = Lx / abs(coeffs[0]) if coeffs[0] != 0 else float('inf')
    print(f"  Linear fit: v = {coeffs[0]:.4f} Lx/t, period = {period:.1f}T")
