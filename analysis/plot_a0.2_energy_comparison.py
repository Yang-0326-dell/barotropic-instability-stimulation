"""
Create composite energy figure for a=0.2 with k=1,2,3,4,rand
Arrange 5 energy plots in 2 columns x 3 rows
"""
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

energy_dir = 'energy_png'

# Energy PNG files for a=0.2
energy_files = {
    'k=1': os.path.join(energy_dir, 'E_a0.2_ap0.2_k1.png'),
    'k=2': os.path.join(energy_dir, 'E_a0.2_ap0.2_k2.png'),
    'k=3': os.path.join(energy_dir, 'E_a0.2_ap0.2_k3.png'),
    'k=4': os.path.join(energy_dir, 'E_a0.2_ap0.2_k4.png'),
    'rand': os.path.join(energy_dir, 'E_a0.2_rand2d_k0-10.png'),
}

# Create figure with 2 columns x 3 rows
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (label, fpath) in enumerate(energy_files.items()):
    if os.path.exists(fpath):
        img = imread(fpath)
        axes[idx].imshow(img)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, f'{label}\n(File not found)', 
                       ha='center', va='center', fontsize=12,
                       transform=axes[idx].transAxes)
        axes[idx].set_title(label, fontsize=12)
        axes[idx].axis('off')

# Empty last subplot
axes[5].axis('off')

plt.suptitle('Perturbation Kinetic Energy (a=0.2)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('vortex_tracking/results/a0.2_energy_comparison.png', dpi=150, bbox_inches='tight')
print("Saved a0.2_energy_comparison.png")
plt.close()
