#!/usr/bin/env python3
"""Plot wave speeds for a=0.2, k=1/2/3/4 in a single figure."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("vortex_tracking/results")

cases = [1, 2, 3, 4]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

for k, color in zip(cases, colors):
    data = np.load(RESULTS_DIR / f'a0.2_k{k}_streamline_tracked.npz')
    t = data['t']
    x_unwrap = data['x_unwrap']
    v = np.gradient(x_unwrap, t)
    v_mean = np.mean(v)
    
    axes[0].plot(t, x_unwrap, 'o-', color=color, label=f'k={k}', markersize=4, linewidth=1.5)
    axes[1].plot(t, v, '-', color=color, label=f"k={k}: {v_mean:.3f}", linewidth=1.5)

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
plt.savefig('composite_figures/wave_speed_a0.2_k1234_combined.png', dpi=150, bbox_inches='tight')
print("Saved wave_speed_a0.2_k1234_combined.png")
plt.close()
