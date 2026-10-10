#!/usr/bin/env python3
"""通过流线密度找涡旋中心"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 配置
tracking_dir = Path("/Users/coalminer/Desktop/流体模拟/HW1/vortex_tracking")
results_dir = tracking_dir / "results"
results_dir.mkdir(exist_ok=True)

# 物理参数
Lx = 10
T = 50
time_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

def find_vortex_by_density(frame_path, search_right=True):
    """通过流线密度找涡旋中心"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 阈值化：黑色流线
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 在 y=0 附近取一个窄带（涡旋中心附近）
    y_center = h // 2
    band_height = int(h * 0.08)  # 8% 的高度
    band = binary[y_center-band_height:y_center+band_height, :]
    
    # 计算每列的流线密度（黑色像素数）
    column_density = np.sum(band, axis=0)
    
    # 平滑
    kernel_size = 51
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(column_density, kernel, mode='same')
    
    # 找峰值
    if search_right:
        # 右半部分
        search_start = int(w * 0.3)
        search_region = smoothed[search_start:]
        peak_idx = np.argmax(search_region) + search_start
    else:
        search_end = int(w * 0.5)
        search_region = smoothed[:search_end]
        peak_idx = np.argmax(search_region)
    
    # 涡旋中心 x 坐标（像素）
    cx = peak_idx
    cy = y_center
    
    # 计算密度值（用于评估可靠性）
    density = smoothed[cx]
    
    return (cx, cy, density, smoothed)

# 存储结果
results = {}

# 处理每个 a 值
a_values = ['a0.05', 'a0.08', 'a0.1', 'a0.2']

for a_val in a_values:
    print(f"\n处理 {a_val}...")
    frames_dir = tracking_dir / a_val
    
    positions = []
    
    for t_val in time_steps:
        frame_path = frames_dir / f"t{t_val:.0f}.png"
        if not frame_path.exists():
            print(f"  t={t_val}: 帧不存在")
            continue
        
        # 找右涡旋中心
        result = find_vortex_by_density(frame_path, search_right=True)
        
        if result is not None:
            cx, cy, density, _ = result
            positions.append((t_val, cx, cy, density))
            print(f"  t={t_val}: 涡旋中心 x={cx}, 密度={density:.0f}")
        else:
            print(f"  t={t_val}: 未找到涡旋")
    
    results[a_val] = positions

# 保存结果
for a_val, positions in results.items():
    if positions:
        pos_array = np.array([(p[0], p[1], p[2]) for p in positions])
        np.savez(results_dir / f"{a_val}_vortex_positions.npz",
                 positions=pos_array)

print("\n结果已保存到", results_dir)

# 可视化一个例子
print("\n可视化 a0.05 t=30 的密度分布...")
frame_path = tracking_dir / "a0.05" / "t30.png"
result = find_vortex_by_density(frame_path, search_right=True)
if result:
    cx, cy, density, smoothed = result
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    img = cv2.imread(str(frame_path))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].axvline(cx, color='red', linewidth=2, label=f'检测到的涡旋中心 x={cx}')
    axes[0].set_title(f'a=0.05, t=30')
    axes[0].legend()
    
    axes[1].plot(smoothed)
    axes[1].axvline(cx, color='red', linewidth=2, label=f'峰值 x={cx}')
    axes[1].set_xlabel('x 像素位置')
    axes[1].set_ylabel('流线密度')
    axes[1].set_title('y=0 附近的流线密度分布')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'density_example.png', dpi=100)
    print("保存可视化到", results_dir / 'density_example.png')
