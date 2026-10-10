#!/usr/bin/env python3
"""Plot wave speeds for a=0.2, k=1/2/3/4."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("vortex_tracking/results")

cases = [1, 2, 3, 4]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

wave_speeds = {}

for idx, k in enumerate(cases):
    data = np.load(RESULTS_DIR / f'a0.2_k{k}_streamline_tracked.npz')
    t = data['t']
    x_unwrap = data['x_unwrap']
    
    # Calculate velocity using central difference
    v = np.gradient(x_unwrap, t)
    wave_speed = np.mean(v)
    wave_speeds[k] = wave_speed
    
    ax = axes[idx]
    ax.plot(t, x_unwrap, 'o-', color='blue', linewidth=2, markersize=4, label='Position (unwrapped)')
    ax2 = ax.twinx()
    ax2.plot(t, v, '-', color='red', linewidth=1.5, alpha=0.7, label='Velocity')
    
    ax.set_xlabel('t', fontsize=11)
    ax.set_ylabel('x (unwrapped)', fontsize=11, color='blue')
    ax2.set_ylabel('Velocity', fontsize=11, color='red')
    ax.set_title(f'k={k}, wave speed = {wave_speed:.3f} Lx/t', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('composite_figures/wave_speed_a0.2_k1234.png', dpi=150, bbox_inches='tight')
print("Saved wave_speed_a0.2_k1234.png")
plt.close()

print("\n=== Wave Speed Summary for a=0.2 ===")
print(f"{'k':<5} {'Wave Speed (Lx/t)':<20} {'Period (T)':<15}")
for k in cases:
    period = 10.0 / abs(wave_speeds[k]) if wave_speeds[k] != 0 else float('inf')
    print(f"{k:<5} {wave_speeds[k]:<20.4f} {period:<15.1f}")
