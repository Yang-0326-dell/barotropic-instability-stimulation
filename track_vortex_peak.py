"""
Improved vortex tracking using peak detection and energy growth measurement.
Track vortex center at peak of |q'|, not centroid.
Measure exponential energy growth rate from total perturbation signal.
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

def get_red_profile(frame_bgr, band_half=15):
    """Get column-wise red (positive q') strength near y=0 with narrow band."""
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
    
    kernel = np.ones(5) / 5
    profile = np.convolve(profile, kernel, mode='same')
    return profile

def track_vortex_centroid(profile, prev_x_pix, search_lx=0.8):
    """Track vortex centroid with narrow band for core focus."""
    if prev_x_pix is None:
        return None
    
    search_pix = int(search_lx * PIX_PER_LX)
    domain_width = X_PIX_MAX - X_PIX_MIN
    
    search_left = int(prev_x_pix - search_pix)
    search_right = int(prev_x_pix + search_pix)
    
    # Handle periodic boundary
    offset = 0
    if search_left < X_PIX_MIN:
        profile_ext = np.concatenate([profile[X_PIX_MIN - (domain_width - search_left):], profile])
        offset = domain_width - (X_PIX_MIN - (domain_width - search_left))
        search_left += offset
        search_right += offset
    elif search_right > X_PIX_MAX:
        profile_ext = np.concatenate([profile, profile[:search_right - X_PIX_MAX]])
    else:
        profile_ext = profile
    
    if search_right > len(profile_ext):
        search_right = len(profile_ext)
    
    region = profile_ext[search_left:search_right]
    if len(region) == 0 or region.sum() < 20:
        return None
    
    # Centroid
    x_local = (np.arange(len(region)) * region).sum() / region.sum()
    x_pix = search_left + x_local - offset
    
    # Wrap back to domain
    while x_pix >= X_PIX_MAX:
        x_pix -= domain_width
    while x_pix < X_PIX_MIN:
        x_pix += domain_width
    
    return x_pix

def compute_energy_signal(frame_bgr, band_half=100):
    """Compute total perturbation energy signal (sum of |q'|)."""
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
    band_half = max(25, int(a_val * 500))
    
    print("  Scanning for initial vortex...")
    prev_x_pix = None
    init_frame = 0
    y_center = (Y_PIX_MIN + Y_PIX_MAX) // 2
    
    for frame_idx in range(N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        profile = get_red_profile(frame, band_half=band_half)
        mid = (X_PIX_MIN + X_PIX_MAX) // 2
        region = profile[mid:X_PIX_MAX]
        if region.sum() > 500:
            peak_idx = np.argmax(region)
            prev_x_pix = mid + peak_idx
            init_frame = frame_idx
            t_init = frame_idx * dt_frame
            print(f"  Found at frame {frame_idx} (t={t_init:.1f}), x={pix_to_x(prev_x_pix):.2f}")
            break
    
    if prev_x_pix is None:
        print("  Could not find initial vortex")
        cap.release()
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    
    all_x_pix = [prev_x_pix]
    all_t = [init_frame * dt_frame]
    all_energy = []
    
    for i in range(init_frame, N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        
        t = i * dt_frame
        profile = get_red_profile(frame, band_half=band_half)
        energy = compute_energy_signal(frame, band_half=band_half)
        all_energy.append(energy)
        
        if i > init_frame:
            x_pix = track_vortex_centroid(profile, prev_x_pix, search_lx=0.8)
            if x_pix is not None:
                all_x_pix.append(x_pix)
                all_t.append(t)
                prev_x_pix = x_pix
    
    cap.release()
    
    x_domain = np.array([pix_to_x(px) for px in all_x_pix])
    t_arr = np.array(all_t)
    energy_arr = np.array(all_energy[:len(t_arr)])
    
    t_sub = []
    x_sub = []
    e_sub = []
    for ti, xi, ei in zip(t_arr, x_domain, energy_arr):
        t_int = round(ti)
        if not t_sub or abs(t_int - t_sub[-1]) > 0.5:
            t_sub.append(t_int)
            x_sub.append(xi)
            e_sub.append(ei)
    
    results[a_val] = {'t': np.array(t_sub), 'x': np.array(x_sub)}
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
plt.savefig('vortex_tracking/results/vortex_tracking_continuous.png', dpi=150, bbox_inches='tight')
print("\nSaved vortex_tracking_continuous.png")
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
plt.savefig('vortex_tracking/results/energy_growth.png', dpi=150, bbox_inches='tight')
print("\nSaved energy_growth.png")
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
