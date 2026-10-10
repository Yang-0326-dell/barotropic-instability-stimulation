#!/usr/bin/env python3
"""Plot vortex center position and velocity vs time for all four cases"""
import numpy as np
import matplotlib.pyplot as plt

# Manual measurements from visual inspection of frames
# Right vortex center x-position (in physical units, Lx=10)
data = {
    0.05: {
        't': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'x': [6.5, 4.0, 8.0, 6.0, 4.0, 7.0, 8.5, 6.0, 3.5, 1.0],
    },
    0.08: {
        't': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'x': [7.0, 5.0, 8.0, 6.0, 4.5, 7.0, 8.0, 6.0, 4.0, 2.0],
    },
    0.10: {
        't': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'x': [7.0, 5.0, 8.0, 6.0, 4.0, 4.0, 7.0, 5.0, 3.0, 1.0],
    },
    0.20: {
        't': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'x': [7.0, 5.0, 7.5, 6.0, 5.0, 6.5, 7.5, 6.0, 5.0, 4.0],
    },
}

colors = {0.05: '#1f77b4', 0.08: '#ff7f0e', 0.10: '#2ca02c', 0.20: '#d62728'}
markers = {0.05: 'o', 0.08: 's', 0.10: '^', 0.20: 'D'}

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: position vs time
ax1 = axes[0]
for a_val, d in data.items():
    ax1.plot(d['t'], d['x'], marker=markers[a_val], color=colors[a_val],
             label=f'a = {a_val}', linewidth=2, markersize=8)
ax1.set_ylabel('Vortex center position $x$ (Lx)', fontsize=14)
ax1.set_title('Right vortex center position vs time', fontsize=15)
ax1.legend(fontsize=12, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, 10.5)

# Bottom: velocity vs time (finite difference)
ax2 = axes[1]
for a_val, d in data.items():
    t = np.array(d['t'])
    x = np.array(d['x'])
    # Unwrap periodic positions
    x_unwrap = np.unwrap(x, period=10)
    v = np.diff(x_unwrap) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2
    ax2.plot(t_mid, v, marker=markers[a_val], color=colors[a_val],
             label=f'a = {a_val}', linewidth=2, markersize=8)
ax2.set_xlabel('Time $t$ (T)', fontsize=14)
ax2.set_ylabel('Propagation speed $v_x$ (Lx/T)', fontsize=14)
ax2.set_title('Vortex propagation speed vs time', fontsize=15)
ax2.legend(fontsize=12, loc='best')
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('/Users/coalminer/Desktop/流体模拟/HW1/vortex_tracking/results/vortex_speed_vs_time.png', dpi=150)
print("Saved vortex_speed_vs_time.png")

# Also save a combined single-panel version
fig2, ax = plt.subplots(figsize=(10, 6))
for a_val, d in data.items():
    t = np.array(d['t'])
    x = np.array(d['x'])
    x_unwrap = np.unwrap(x, period=10)
    v = np.diff(x_unwrap) / np.diff(t)
    t_mid = (t[:-1] + t[1:]) / 2
    ax.plot(t_mid, v, marker=markers[a_val], color=colors[a_val],
            label=f'a = {a_val}', linewidth=2, markersize=8)
ax.set_xlabel('Time $t$ (T)', fontsize=14)
ax.set_ylabel('Propagation speed $v_x$ (Lx/T)', fontsize=14)
ax.set_title('Right vortex propagation speed vs time', fontsize=15)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.savefig('/Users/coalminer/Desktop/流体模拟/HW1/vortex_tracking/results/vortex_speed.png', dpi=150)
print("Saved vortex_speed.png")
