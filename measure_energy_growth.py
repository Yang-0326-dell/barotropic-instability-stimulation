import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

ENERGY_DIR = 'energy_png'

EXPERIMENTS = {
    0.05: f'{ENERGY_DIR}/E_a0.05_ap0.05_k1.png',
    0.08: f'{ENERGY_DIR}/E_a0.08_ap0.08_k1.png',
    0.10: f'{ENERGY_DIR}/E_a0.1_ap0.1_k1.png',
    0.20: f'{ENERGY_DIR}/E_a0.2_ap0.2_k1.png',
}

# Two-segment fit ranges for each experiment
FIT_RANGES = {
    0.05: [(3, 5), (7, 10)],
    0.08: [(3, 5), (7, 13)],
    0.10: [(3, 5), (8, 15)],
    0.20: [(4, 7), (20, 50)],
}
Y_LOG_TOP = -2
Y_LOG_BOT = -8
X_MIN, X_MAX = 0, 50

def detect_plot_area(arr):
    """Detect plot area from black axis lines."""
    h, w = arr.shape[:2]
    gray = np.mean(arr[:, :, :3], axis=2)
    black_mask = gray < 80
    col_black = np.sum(black_mask, axis=0)
    row_black = np.sum(black_mask, axis=1)
    
    left = right = top = bottom = None
    for c in range(w):
        if col_black[c] > h * 0.3:
            left = c; break
    for c in range(w - 1, -1, -1):
        if col_black[c] > h * 0.3:
            right = c; break
    for r in range(h):
        if row_black[r] > w * 0.3:
            top = r; break
    for r in range(h - 1, -1, -1):
        if row_black[r] > w * 0.3:
            bottom = r; break
    return left, right, top, bottom

def extract_curve(arr, px_left, px_right, px_top, px_bottom):
    """Extract blue curve center positions."""
    r, g, b = arr[:, :, 0].astype(int), arr[:, :, 1].astype(int), arr[:, :, 2].astype(int)
    blue_mask = (b > 150) & (b > r + 30) & (b > g + 10) & (r < 150)
    
    t_vals, log_e_vals = [], []
    for px in range(px_left + 2, px_right - 1):
        col_blue = np.where(blue_mask[px_top + 1:px_bottom, px])[0]
        if len(col_blue) > 0:
            center_row = np.median(col_blue) + px_top + 1
            t = X_MIN + (px - px_left) / (px_right - px_left) * (X_MAX - X_MIN)
            log_e = Y_LOG_TOP - (center_row - px_top) / (px_bottom - px_top) * (Y_LOG_TOP - Y_LOG_BOT)
            t_vals.append(t)
            log_e_vals.append(log_e)
    return np.array(t_vals), np.array(log_e_vals)

def fit_segment(t, log_e, t_start, t_end):
    """Fit log10(E) = slope * t + c in a fixed time range.
    Returns sigma (natural log growth rate) and log10 slope."""
    mask = (t >= t_start) & (t <= t_end)
    if mask.sum() < 5:
        return np.nan, np.nan
    t_fit = t[mask]
    e_fit = log_e[mask]
    coeffs = np.polyfit(t_fit, e_fit, 1)
    log10_slope = coeffs[0]
    intercept = coeffs[1]
    sigma = log10_slope * np.log(10)
    return sigma, log10_slope, intercept, t_fit, e_fit

# Process
results = {}
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

colors_seg = ['red', 'orange']

for idx, (a_val, img_path) in enumerate(EXPERIMENTS.items()):
    print(f"\n=== a={a_val} ===")
    img = Image.open(img_path)
    arr = np.array(img)
    
    px_left, px_right, px_top, px_bottom = detect_plot_area(arr)
    print(f"  Plot area: left={px_left}, right={px_right}, top={px_top}, bottom={px_bottom}")
    
    t, log_e = extract_curve(arr, px_left, px_right, px_top, px_bottom)
    print(f"  Extracted {len(t)} points, t=[{t.min():.1f},{t.max():.1f}], log10(E)=[{log_e.min():.2f},{log_e.max():.2f}]")
    
    seg_results = []
    for seg_idx, (t_start, t_end) in enumerate(FIT_RANGES[a_val]):
        sigma, log10_slope, intercept, t_fit, e_fit = fit_segment(t, log_e, t_start, t_end)
        seg_results.append({
            'sigma': sigma, 'log10_slope': log10_slope, 'intercept': intercept,
            't_fit': t_fit, 'e_fit': e_fit, 't_start': t_start, 't_end': t_end
        })
        print(f"  Segment {seg_idx+1} t∈[{t_start},{t_end}]: σ={sigma:.4f}, log10 slope={log10_slope:.4f}")
    
    results[a_val] = {'t': t, 'log_e': log_e, 'segments': seg_results}
    
    ax = axes[idx]
    ax.plot(t, log_e, 'b.', markersize=2, alpha=0.4, label='Extracted')
    for seg_idx, seg in enumerate(seg_results):
        ax.plot(seg['t_fit'], seg['e_fit'], '.', color=colors_seg[seg_idx],
                markersize=3, alpha=0.6, label=f'Seg{seg_idx+1} [{seg["t_start"]},{seg["t_end"]}]')
        t_line = np.array([seg['t_fit'].min(), seg['t_fit'].max()])
        ax.plot(t_line, seg['log10_slope'] * t_line + seg['intercept'],
                '-', color=colors_seg[seg_idx], linewidth=2,
                label=f"σ{seg_idx+1}={seg['sigma']:.4f}")
    ax.set_title(f'a={a_val}, k=1', fontsize=12)
    ax.set_xlabel('t')
    ax.set_ylabel('log₁₀(E)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('energy_growth_rates.png', dpi=150, bbox_inches='tight')
print("\nSaved energy_growth_rates.png")
plt.close()

print("\n=== Growth Rate Summary (Two Segments) ===")
print(f"{'a':<10} {'σ1 (seg1)':<15} {'σ2 (seg2)':<15} {'seg1 range':<15} {'seg2 range':<15}")
for a_val, res in results.items():
    s1, s2 = res['segments']
    print(f"{a_val:<10} {s1['sigma']:<15.4f} {s2['sigma']:<15.4f} [{s1['t_start']},{s1['t_end']}]".ljust(35) + f"[{s2['t_start']},{s2['t_end']}]")
