"""
Composite figure for a=0.1:
- Pert/Total frames at t=0, 10, 20, 50 (4 rows x 2 columns)
- Vortex position vs T (bottom left)
- Wave speed vs T (bottom right)
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Frame paths
pert_frames = {
    0: '/Users/coalminer/Downloads/frames_dt1/solve_2D_nonlinear_case1_ap0.1_k1_pert_t00.png',
    10: '/Users/coalminer/Downloads/frames_dt1/solve_2D_nonlinear_case1_ap0.1_k1_pert_t10.png',
    20: '/Users/coalminer/Downloads/frames_dt1/solve_2D_nonlinear_case1_ap0.1_k1_pert_t20.png',
    50: '/Users/coalminer/Downloads/frames_dt1/solve_2D_nonlinear_case1_ap0.1_k1_pert_t50.png'
}

total_frames = {
    0: '/Users/coalminer/Downloads/frames_dt1_total/solve_2D_nonlinear_case1_ap0.1_k1_total_t00.png',
    10: '/Users/coalminer/Downloads/frames_dt1_total/solve_2D_nonlinear_case1_ap0.1_k1_total_t10.png',
    20: '/Users/coalminer/Downloads/frames_dt1_total/solve_2D_nonlinear_case1_ap0.1_k1_total_t20.png',
    50: '/Users/coalminer/Downloads/frames_dt1_total/solve_2D_nonlinear_case1_ap0.1_k1_total_t50.png'
}

# Load new tracking data
track_data = np.load('vortex_tracking/results/a0.1_tracked.npz')
t_track = track_data['t']
x_track = track_data['x']

# Unwrap position
x_unwrap = np.unwrap(x_track, period=10)

# Calculate velocity from unwrapped position
v_track = np.diff(x_unwrap) / np.diff(t_track)
t_mid = (t_track[:-1] + t_track[1:]) / 2

# Create figure
fig = plt.figure(figsize=(16, 20))

# Time points
times = [0, 10, 20, 50]

# Plot frames in 4x2 grid
for i, t in enumerate(times):
    # Perturbation frame (left column)
    ax_pert = plt.subplot(5, 2, i*2 + 1)
    img_pert = Image.open(pert_frames[t])
    ax_pert.imshow(img_pert)
    ax_pert.set_title(f'Perturbation t={t}', fontsize=14)
    ax_pert.axis('off')

    # Total frame (right column)
    ax_total = plt.subplot(5, 2, i*2 + 2)
    img_total = Image.open(total_frames[t])
    ax_total.imshow(img_total)
    ax_total.set_title(f'Total t={t}', fontsize=14)
    ax_total.axis('off')

# Vortex position vs T (bottom left) - showing unwrapped position
ax_pos = plt.subplot(5, 2, 9)
ax_pos.plot(t_track, x_unwrap, 'b-o', linewidth=2, markersize=4)
ax_pos.set_xlabel('Time (t)', fontsize=12)
ax_pos.set_ylabel('Unwrapped x-position (Lx)', fontsize=12)
ax_pos.set_title('Vortex Position vs Time (Unwrapped)', fontsize=14)
ax_pos.grid(True, alpha=0.3)

# Wave speed vs T (bottom right) - using unwrapped method
ax_vel = plt.subplot(5, 2, 10)

# Smooth velocity with moving average
window = 5
v_smooth = np.convolve(v_track, np.ones(window)/window, mode='valid')
t_smooth = (t_mid[:len(v_smooth)] + t_mid[-len(v_smooth):]) / 2

ax_vel.plot(t_mid, v_track, 'r-', linewidth=1, alpha=0.3, label='Raw')
ax_vel.plot(t_smooth, v_smooth, 'r-o', linewidth=2, markersize=4, label='Smoothed')
ax_vel.set_xlabel('Time (t)', fontsize=12)
ax_vel.set_ylabel('Velocity (Lx/t)', fontsize=12)
ax_vel.set_title('Wave Speed vs Time (Unwrapped)', fontsize=14)
ax_vel.grid(True, alpha=0.3)
ax_vel.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# Add linear fit
coeffs = np.polyfit(t_track, x_unwrap, 1)
v_fit = coeffs[0]
ax_pos.axhline(y=x_unwrap[0] + v_fit * (t_track[-1] - t_track[0]),
               color='g', linestyle='--', alpha=0.5, label=f'Linear fit: v={v_fit:.3f}')
ax_pos.legend(fontsize=10)

# Add mean velocity to velocity plot
v_mean = v_track[np.abs(t_mid - 25) < 20].mean()  # average over middle portion
ax_vel.axhline(y=v_mean, color='g', linestyle='--', alpha=0.5, label=f'Mean v={v_mean:.3f}')
ax_vel.legend(fontsize=10)

plt.tight_layout()
plt.savefig('vortex_tracking/results/a0.1_comparison.png', dpi=150, bbox_inches='tight')
print("Figure saved to vortex_tracking/results/a0.1_comparison.png")
plt.close()

# Print summary
print(f"\n=== a=0.1 Tracking Summary ===")
print(f"Samples: {len(t_track)}")
print(f"Time range: [{t_track[0]:.1f}, {t_track[-1]:.1f}]")
print(f"Position range (periodic): [{x_track.min():.2f}, {x_track.max():.2f}]")
print(f"Position range (unwrapped): [{x_unwrap.min():.2f}, {x_unwrap.max():.2f}]")
print(f"Linear fit velocity: {v_fit:.4f} Lx/t")
print(f"Period: {10/abs(v_fit):.1f} T")
print(f"Mean velocity (t>10): {v_track[t_mid>10].mean():.4f} Lx/t")
