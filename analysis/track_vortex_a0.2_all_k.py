"""
Track vortex using black streamline curvature detection for a=0.2, k=1/2/3/4.
Same method as track_vortex_streamline.py
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
    1: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4"),
    2: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k2_pert.mp4"),
    3: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k3_pert.mp4"),
    4: os.path.join(DL, "solve_2D_nonlinear_case1_abg0.2_ap0.2_k4_pert.mp4"),
}

def find_vortex_center_streamline(frame_bgr, prev_x_pix, prev_y_pix, search_lx=1.0):
    """Find vortex center by detecting high streamline curvature region."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    black_mask = (gray < 100).astype(np.uint8)
    
    h, w = black_mask.shape
    crop = np.zeros((Y_PIX_MAX - Y_PIX_MIN, X_PIX_MAX - X_PIX_MIN), dtype=np.uint8)
    crop[:] = black_mask[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]
    
    search_pix = int(search_lx * PIX_PER_LX)
    domain_width_pix = X_PIX_MAX - X_PIX_MIN
    
    prev_x_local = prev_x_pix - X_PIX_MIN
    search_left = int(prev_x_local - search_pix)
    search_right = int(prev_x_local + search_pix)
    
    crop_ext = np.concatenate([crop, crop[:, :search_pix + 50]], axis=1)
    
    if search_left < 0:
        search_left += domain_width_pix
        search_right += domain_width_pix
    
    if search_right > crop_ext.shape[1]:
        search_right = crop_ext.shape[1]
    
    search_top = max(0, int((prev_y_pix - Y_PIX_MIN) - search_pix * 0.3))
    search_bottom = min(crop.shape[0], int((prev_y_pix - Y_PIX_MIN) + search_pix * 0.3))
    
    region = crop_ext[search_top:search_bottom, search_left:search_right]
    
    if region.sum() < 50:
        return None, None
    
    kernel_size = 40
    sigma = kernel_size / 4
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel_2d = kernel @ kernel.T
    
    density = cv2.filter2D(region.astype(np.float32), -1, kernel_2d)
    
    max_loc = np.unravel_index(np.argmax(density), density.shape)
    max_val = density[max_loc]
    
    if max_val < 0.05:
        return None, None
    
    cy_max, cx_max = max_loc
    refine_size = 100
    
    y1 = max(0, cy_max - refine_size // 2)
    y2 = min(region.shape[0], cy_max + refine_size // 2)
    x1 = max(0, cx_max - refine_size // 2)
    x2 = min(region.shape[1], cx_max + refine_size // 2)
    
    refine_region = region[y1:y2, x1:x2]
    
    if refine_region.sum() < 20:
        cx_final = cx_max
        cy_final = cy_max
    else:
        ys, xs = np.where(refine_region > 0)
        if len(xs) == 0:
            cx_final = cx_max
            cy_final = cy_max
        else:
            cx_final = xs.mean()
            cy_final = ys.mean()
    
    cx = search_left + x1 + cx_final + X_PIX_MIN
    cy = search_top + y1 + cy_final + Y_PIX_MIN
    
    while cx >= X_PIX_MAX:
        cx -= domain_width_pix
    while cx < X_PIX_MIN:
        cx += domain_width_pix
    
    return cx, cy


results = {}

for k_val, vpath in VIDEOS.items():
    print(f"\n=== k={k_val} ===")
    if not os.path.exists(vpath):
        print(f"  Video not found: {vpath}")
        continue
    
    cap = cv2.VideoCapture(vpath)
    
    print("  Scanning for initial vortex...")
    prev_x_pix = None
    prev_y_pix = None
    init_frame = 0
    
    for frame_idx in range(501):
        ok, frame = cap.read()
        if not ok:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_mask = (gray < 100).astype(np.uint8)
        
        h, w = black_mask.shape
        crop = np.zeros((Y_PIX_MAX - Y_PIX_MIN, X_PIX_MAX - X_PIX_MIN), dtype=np.uint8)
        crop[:] = black_mask[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]
        
        kernel_size = 40
        sigma = kernel_size / 4
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_2d = kernel @ kernel.T
        density = cv2.filter2D(crop.astype(np.float32), -1, kernel_2d)
        
        mid = crop.shape[1] // 2
        right_half = density[:, mid:]
        
        max_loc = np.unravel_index(np.argmax(right_half), right_half.shape)
        max_val = right_half[max_loc]
        
        if max_val > 0.05:
            prev_x_pix = X_PIX_MIN + mid + max_loc[1]
            prev_y_pix = Y_PIX_MIN + max_loc[0]
            init_frame = frame_idx
            t_init = frame_idx * dt_frame
            print(f"  Found at frame {frame_idx} (t={t_init:.1f}), x={pix_to_x(prev_x_pix):.2f}, y={pix_to_y(prev_y_pix):.2f}")
            break
    
    if prev_x_pix is None:
        print("  Could not find initial vortex")
        cap.release()
        continue
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)
    
    all_x_pix = [prev_x_pix]
    all_y_pix = [prev_y_pix]
    all_t = [init_frame * dt_frame]
    
    for i in range(init_frame + 1, N_frames):
        ok, frame = cap.read()
        if not ok:
            break
        
        t = i * dt_frame
        x_pix, y_pix = find_vortex_center_streamline(frame, prev_x_pix, prev_y_pix, search_lx=1.0)
        
        if x_pix is not None and y_pix is not None:
            dx = abs(x_pix - prev_x_pix)
            domain_width = X_PIX_MAX - X_PIX_MIN
            if dx > domain_width / 2:
                dx = domain_width - dx
            
            if dx < 1.2 * PIX_PER_LX:
                all_x_pix.append(x_pix)
                all_y_pix.append(y_pix)
                all_t.append(t)
                prev_x_pix = x_pix
                prev_y_pix = y_pix
    
    cap.release()
    
    x_domain = np.array([pix_to_x(px) for px in all_x_pix])
    y_domain = np.array([pix_to_y(py) for py in all_y_pix])
    t_arr = np.array(all_t)
    
    t_sub = []
    x_sub = []
    y_sub = []
    for ti, xi, yi in zip(t_arr, x_domain, y_domain):
        t_int = round(ti)
        if not t_sub or abs(t_int - t_sub[-1]) > 0.5:
            t_sub.append(t_int)
            x_sub.append(xi)
            y_sub.append(yi)
    
    t_arr = np.array(t_sub)
    x_arr = np.array(x_sub)
    y_arr = np.array(y_sub)
    
    x_unwrap = np.unwrap(x_arr, period=Lx)
    v_central = np.gradient(x_unwrap, t_arr)
    v_mean = np.nanmean(v_central)
    
    results[k_val] = {'t': t_arr, 'x': x_arr, 'y': y_arr, 'x_unwrap': x_unwrap, 
                      'v_central': v_central, 'v_mean': v_mean}
    print(f"  {len(t_sub)} samples at dt=1")
    print(f"  Mean velocity (central diff): {v_mean:.4f} Lx/t")
    print(f"  Period: {Lx / abs(v_mean):.1f} T" if v_mean != 0 else "  Period: inf")

# Save
for k_val, res in results.items():
    np.savez(f'vortex_tracking/results/a0.2_k{k_val}_streamline_tracked.npz', 
             t=res['t'], x=res['x'], y=res['y'], x_unwrap=res['x_unwrap'])

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red'}

for k_val, res in results.items():
    t = res['t']
    x_unwrap = res['x_unwrap']
    v_central = res['v_central']
    
    axes[0].plot(t, x_unwrap, 'o-', color=colors[k_val], label=f'k={k_val}', markersize=4, linewidth=1.5)
    axes[1].plot(t, v_central, '-', color=colors[k_val], label=f"k={k_val}: {res['v_mean']:.3f}", linewidth=1.5)

axes[0].set_title('Unwrapped Vortex Position from Black Streamlines (a=0.2)', fontsize=14)
axes[0].set_ylabel('Unwrapped x (Lx)', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].set_title('Vortex Velocity (Central Difference)', fontsize=14)
axes[1].set_ylabel('Velocity (Lx/t)', fontsize=12)
axes[1].set_xlabel('Time (t)', fontsize=12)
axes[1].set_xlim(0, 50)
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('vortex_tracking/results/vortex_tracking_a0.2_k1234.png', dpi=150, bbox_inches='tight')
print("\nSaved vortex_tracking_a0.2_k1234.png")
plt.close()

print("\n=== Final Velocity Summary ===")
print(f"{'k':<10} {'v_mean':<15} {'period':<15}")
for k_val, res in results.items():
    period = Lx / abs(res['v_mean']) if res['v_mean'] != 0 else float('inf')
    print(f"{k_val:<10} {res['v_mean']:<15.4f} {period:<15.1f}")
