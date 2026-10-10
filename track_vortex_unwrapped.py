"""
Track ONE vortex continuously through periodic domain using vortex_peak data.
The vortex_peak tracker finds max |q'| in right half (x>5), but jumps between
the two vortices as they move. We fix this by:
1. Detecting jumps (|dx| > 3 in one timestep)
2. Adding/subtracting 10 to maintain continuity
3. Then unwrapping for the periodic domain
"""
import numpy as np
import matplotlib.pyplot as plt

cases = {
    0.05: 'wave_speed/a0.05_vortex_peak.npz',
    0.08: 'wave_speed/a0.08_vortex_peak.npz',
    0.10: 'wave_speed/a0.1_vortex_peak.npz',
    0.20: 'wave_speed/a0.2_vortex_peak.npz',
}

Lx = 10.0

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
colors = {0.05: 'blue', 0.08: 'green', 0.10: 'orange', 0.20: 'red'}
labels = {0.05: 'a=0.05', 0.08: 'a=0.08', 0.10: 'a=0.10', 0.20: 'a=0.20'}

all_results = {}

for a_val, fname in cases.items():
    d = np.load(fname)
    t = d['t']
    x_raw = d['x']
    
    # Step 1: Fix jumps in the raw data
    # The tracker jumps between two vortices (~5 apart). When |dx| > 3, it's a jump.
    x_fixed = [x_raw[0]]
    for i in range(1, len(x_raw)):
        dx = x_raw[i] - x_raw[i-1]
        if abs(dx) > 3:  # jump detected
            # The other vortex is ~5 away. If we jumped from right to left vortex,
            # add 10; if left to right, subtract 10.
            if dx < 0:  # jumped left (right vortex -> left vortex)
                x_fixed.append(x_raw[i] + Lx)
            else:  # jumped right
                x_fixed.append(x_raw[i] - Lx)
        else:
            x_fixed.append(x_raw[i])
    x_fixed = np.array(x_fixed)
    
    # Step 2: Unwrap for periodic domain
    x_mod = x_fixed % Lx  # bring back to [0, 10]
    x_unwrap = np.unwrap(x_mod, period=Lx)
    
    # Step 3: Compute velocity
    dt_arr = np.diff(t)
    v = np.diff(x_unwrap) / dt_arr
    t_mid = (t[:-1] + t[1:]) / 2
    
    all_results[a_val] = {'t': t, 'x_raw': x_raw, 'x_fixed': x_fixed, 
                          'x_unwrap': x_unwrap, 'v': v, 't_mid': t_mid}
    
    # Plot raw periodic position
    axes[0].plot(t, x_raw, '.', color=colors[a_val], label=labels[a_val], 
                 markersize=2, alpha=0.5)
    
    # Plot unwrapped position
    axes[1].plot(t, x_unwrap, '-', color=colors[a_val], label=labels[a_val], 
                 linewidth=1.5)
    
    # Plot velocity
    axes[2].plot(t_mid, v, '.', color=colors[a_val], label=labels[a_val], 
                 markersize=2, alpha=0.5)

# Panel 1: Raw
axes[0].set_xlabel('Time (t)', fontsize=12)
axes[0].set_ylabel('Vortex x-position (Lx)', fontsize=12)
axes[0].set_title('Raw Vortex Peak Position (periodic)', fontsize=14)
axes[0].set_xlim(0, 50)
axes[0].set_ylim(-0.5, 10.5)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)
axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0].axhline(y=10, color='k', linestyle='--', alpha=0.3)

# Panel 2: Unwrapped
axes[1].set_xlabel('Time (t)', fontsize=12)
axes[1].set_ylabel('Unwrapped x-position (Lx)', fontsize=12)
axes[1].set_title('Continuous Vortex Tracking (Unwrapped)', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

# Panel 3: Velocity
axes[2].set_xlabel('Time (t)', fontsize=12)
axes[2].set_ylabel('Vortex velocity (Lx/t)', fontsize=12)
axes[2].set_title('Vortex Propagation Speed', fontsize=14)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=11)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('vortex_tracking/results/vortex_tracking_v2.png', dpi=150, bbox_inches='tight')
print("Saved vortex_tracking_v2.png")

# Print summary statistics
print("\n=== Vortex velocity summary ===")
for a_val, res in all_results.items():
    v = res['v']
    t_mid = res['t_mid']
    # Use t > 10 to avoid initial transient
    mask = t_mid > 10
    if mask.sum() > 0:
        v_stable = v[mask]
        print(f"a={a_val}: mean v = {v_stable.mean():.4f}, std = {v_stable.std():.4f}, "
              f"range = [{v_stable.min():.3f}, {v_stable.max():.3f}], n={mask.sum()}")
    
    # Linear fit over full range
    coeffs = np.polyfit(res['t'], res['x_unwrap'], 1)
    print(f"  Linear fit: v = {coeffs[0]:.4f} Lx/t, x0 = {coeffs[1]:.2f}")

plt.close()
