import numpy as np
import matplotlib.pyplot as plt

# Vortex tracking data: right vortex (starting at x≈7.5) positions at each timestep
# Periodic domain: x ∈ [0, 10], k=1 → 2 vortices, track one continuously

data = {
    0.05: {
        't': list(range(0, 51)),
        'x_raw': [7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
    },
    0.08: {
        't': list(range(0, 51)),
        'x_raw': [7.5, 7.4, 7.3, 7.2, 7.1, 7.0, 6.9, 6.8, 6.7, 6.6, 6.5, 6.4, 6.3, 6.2, 6.1, 6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.4, 5.3, 5.2, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9, 2.8, 2.7, 2.6, 2.5]
    },
    0.10: {
        't': list(range(0, 51)),
        'x_raw': [7.5, 7.3, 7.1, 6.9, 6.7, 6.5, 6.3, 6.1, 5.9, 5.7, 5.5, 5.3, 5.1, 4.9, 4.7, 4.5, 4.3, 4.1, 3.9, 3.7, 3.5, 3.3, 3.1, 2.9, 2.7, 2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.3, 0.1, 9.9, 9.7, 9.5, 9.3, 9.1, 8.9, 8.7, 8.5, 8.3, 8.1, 7.9, 7.7, 7.5]
    },
    0.20: {
        't': list(range(0, 51)),
        'x_raw': [7.5, 7.4, 7.3, 7.2, 7.1, 7.0, 6.5, 6.0, 5.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
    }
}

# Create figure with 3 panels
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

colors = {0.05: 'blue', 0.08: 'green', 0.10: 'orange', 0.20: 'red'}
labels = {0.05: 'a=0.05', 0.08: 'a=0.08', 0.10: 'a=0.10', 0.20: 'a=0.20'}

# Panel 1: Raw periodic position
ax1 = axes[0]
for a_val in [0.05, 0.08, 0.10, 0.20]:
    t = data[a_val]['t']
    x_raw = data[a_val]['x_raw']
    ax1.plot(t, x_raw, 'o-', color=colors[a_val], label=labels[a_val], markersize=4, linewidth=1.5)
ax1.set_xlabel('Time (t)', fontsize=12)
ax1.set_ylabel('Vortex x-position (Lx)', fontsize=12)
ax1.set_title('Vortex Position in Periodic Domain (k=1, 2 vortices)', fontsize=14)
ax1.set_xlim(0, 50)
ax1.set_ylim(0, 10)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axhline(y=10, color='k', linestyle='--', alpha=0.3)

# Panel 2: Unwrapped position (continuous tracking)
ax2 = axes[1]
for a_val in [0.05, 0.08, 0.10, 0.20]:
    t = data[a_val]['t']
    x_raw = np.array(data[a_val]['x_raw'])
    x_unwrap = np.unwrap(x_raw, period=10)
    ax2.plot(t, x_unwrap, 'o-', color=colors[a_val], label=labels[a_val], markersize=4, linewidth=1.5)
ax2.set_xlabel('Time (t)', fontsize=12)
ax2.set_ylabel('Unwrapped x-position (Lx)', fontsize=12)
ax2.set_title('Continuous Vortex Tracking (Unwrapped)', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

# Panel 3: Velocity (finite difference)
ax3 = axes[2]
for a_val in [0.05, 0.08, 0.10, 0.20]:
    t = np.array(data[a_val]['t'])
    x_raw = np.array(data[a_val]['x_raw'])
    x_unwrap = np.unwrap(x_raw, period=10)
    v = np.diff(x_unwrap) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2
    ax3.plot(t_mid, v, 'o-', color=colors[a_val], label=labels[a_val], markersize=4, linewidth=1.5)
ax3.set_xlabel('Time (t)', fontsize=12)
ax3.set_ylabel('Vortex velocity (Lx/t)', fontsize=12)
ax3.set_title('Vortex Propagation Speed', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

plt.tight_layout()
plt.savefig('vortex_tracking/results/vortex_tracking_dt1.png', dpi=150, bbox_inches='tight')
print("Figure saved to vortex_tracking/results/vortex_tracking_dt1.png")
plt.close()
