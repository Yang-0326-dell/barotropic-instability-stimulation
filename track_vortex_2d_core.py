"""
Track vortex using 2D streamline center method.
Find center of strongest red region (vortex core) in 2D.
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

def pix_to_y(py):
    return 2.0 - (py - Y_PIX_MIN) / (Y_PIX_MAX - Y_PIX_MIN) * 4.0

DL = os.path.expanduser("~/Downloads")
VIDEOS = {
    0.05: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert.mp4"),
    0.08: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert.mp4"),
    0.10: os.path.join(DL, "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4"),
    0.20: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4"),
}

def find_vortex_core_2d(frame_bgr, prev_x_pix=None, search_lx=1.0, top_percentile=90):
    """
    Find vortex core using 2D centroid of strongest red pixels.
    Only use top percentile of red intensity to focus on core.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Crop to plot area
    crop = np.zeros_like(red_mask)
    crop[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX] = red_mask[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]
    
    if prev_x_pix is not None:
        # Restrict to search window
        search_pix = int(search_lx * PIX_PER_LX)
        search_left = int(prev_x_pix - search_pix)
        search_right = int(prev_x_pix + search_pix)
        
        # Handle periodic boundary
        domain_width = X_PIX_MAX - X_PIX_MIN
        if search_left < X_PIX_MIN:
            search_left += domain_width
            search_right += domain_width
        
        # Create masked version
        masked = np.zeros_like(crop)
        if search_right > X_PIX_MAX:
            # Wrap around
            masked[:, X_PIX_MIN:search_right - domain_width] = crop[:, X_PIX_MIN:search_right - domain_width]
            masked[:, search_left - domain_width:X_PIX_MAX] = crop[:, search_left - domain_width:X_PIX_MAX]
        else:
            masked[:, search_left:search_right] = crop[:, search_left:search_right]
    else:
        masked = crop
        # For initial scan, use right half
        mid = (X_PIX_MIN + X_PIX_MAX) // 2
        masked[:, :mid] = 0
    
    # Get coordinates of red pixels
    ys, xs = np.where(masked > 0)
    if len(xs) < 10:
        return None, None
    
    # Use only top percentile by intensity (approximate by using all red pixels)
    # For better accuracy, we could use the actual HSV V channel
    # But for now, binary mask should work
    
    # Compute 2D centroid
    x_centroid = xs.mean()
    y_centroid = ys.mean()
    
    return x_centroid, y_centroid

def compute_energy_signal(frame_bgr, band_half=100):
    """Compute total perturbation energy signal."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (140, 255, 255))
    
    pert = cv2.bitwise_or(red_mask, blue_mask)
    y_center = (Y_PIX_MIN + Y_PIX_MAX) // 2
    band = pert[y_center - band_half:y_center + band_half, X_PIX_MIN:X_PIX_MAX]
    return band.sum()

results = {}
energy_data = {}

for a_val, vpath in VIDEOS.items():
    print(f"\n=== a={a_val} ===")
    if not os.path.exists(vpath):
        print(f"  Video not found: {vpath}")
        continue
    
    cap = cv2.VideoCapture(vpath)
    
    print("  Scanning for initial vortex...")
    prev_x_pix = None
    init_frame = 0
    
    for frame_idx in range(N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        x_pix, y_pix = find_vortex_core_2d(frame, prev_x_pix=None, search_lx=1.0)
        if x_pix is not None and x_pix > (X_PIX_MIN + X_PIX_MAX) / 2:
            prev_x_pix = x_pix
            init_frame = frame_idx
            t_init = frame_idx * dt_frame
            print(f"  Found at frame {frame_idx} (t={t_init:.1f}), x={pix_to_x(prev_x_pix):.2f}, y={pix_to_y(y_pix):.2f}")
            break
    
    if prev_x_pix is None:
        print("  Could not find initial vortex")
        cap.release()
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    
    all_x_pix = [prev_x_pix]
    all_y_pix = []
    all_t = [init_frame * dt_frame]
    all_energy = []
    
    for i in range(init_frame, N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        
        t = i * dt_frame
        energy = compute_energy_signal(frame, band_half=100)
        all_energy.append(energy)
        
        if i > init_frame:
            x_pix, y_pix = find_vortex_core_2d(frame, prev_x_pix, search_lx=1.0)
            if x_pix is not None:
                all_x_pix.append(x_pix)
                all_y_pix.append(y_pix)
                all_t.append(t)
                prev_x_pix = x_pix
    
    cap.release()
    
    x_domain = np.array([pix_to_x(px) for px in all_x_pix])
    y_domain = np.array([pix_to_y(py) for py in all_y_pix]) if all_y_pix else None
    t_arr = np.array(all_t)
    energy_arr = np.array(all_energy[:len(t_arr)])
    
    # Subsample to dt=1
    t_sub = []
    x_sub = []
    y_sub = [] if y_domain is not None else None
    e_sub = []
    for idx, ti in enumerate(t_arr):
        t_int = round(ti)
        if not t_sub or abs(t_int - t_sub[-1]) > 0.5:
            t_sub.append(t_int)
            x_sub.append(x_domain[idx])
            if y_sub is not None:
                y_sub.append(y_domain[idx])
            e_sub.append(energy_arr[idx])
    
    results[a_val] = {'t': np.array(t_sub), 'x': np.array(x_sub)}
    if y_sub is not None:
        results[a_val]['y'] = np.array(y_sub)
    energy_data[a_val] = {'t': np.array(t_sub), 'energy': np.array(e_sub)}
    
    print(f"  {len(t_sub)} samples at dt=1")
    for ti, xi in zip(t_sub[:10], x_sub[:10]):
        print(f"  t={ti:3d}: x={xi:.3f}")

# Save results
for a_val, res in results.items():
    np.savez(f'vortex_tracking/results/a{a_val}_tracked.npz', t=res['t'], x=res['x'])

for a_val, ed in energy_data.items():
    np.savez(f'vortex_tracking/results/a{a_val}_energy.npz', t=ed['t'], energy=ed['energy'])

# Plot tracking results
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
plt.savefig('vortex_tracking/results/vortex_tracking_2d_core.png', dpi=150, bbox_inches='tight')
print("\nSaved vortex_tracking_2d_core.png")
plt.close()

# Plot energy growth
fig, ax = plt.subplots(figsize=(12, 8))
for a_val, ed in energy_data.items():
    t = ed['t']
    energy = ed['energy']
    
    mask = (t > 5) & (t < 30) & (energy > 100)
    if mask.sum() > 5:
        t_fit = t[mask]
        e_fit = energy[mask]
        log_e = np.log(e_fit)
        coeffs = np.polyfit(t_fit, log_e, 1)
        growth_rate = coeffs[0]
        
        ax.semilogy(t, energy, 'o-', color=colors[a_val], label=f'{labels_map[a_val]}: σ={growth_rate:.3f}', 
                    markersize=4, linewidth=1.5, alpha=0.7)
        print(f"a={a_val}: growth rate σ = {growth_rate:.4f} (t={t_fit[0]:.1f} to {t_fit[-1]:.1f})")

ax.set_xlabel('Time (t)', fontsize=12)
ax.set_ylabel('Perturbation Energy (arbitrary units)', fontsize=12)
ax.set_title('Exponential Energy Growth', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('vortex_tracking/results/energy_growth_2d_core.png', dpi=150, bbox_inches='tight')
print("\nSaved energy_growth_2d_core.png")
plt.close()

# Summary
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
