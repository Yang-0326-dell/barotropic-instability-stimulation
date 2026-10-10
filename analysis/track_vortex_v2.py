#!/usr/bin/env python3
"""改进的涡旋跟踪：使用模板匹配"""
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

def create_vortex_template(size=200):
    """创建涡旋模板（同心圆图案）"""
    template = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    for r in range(20, size//2, 15):
        cv2.circle(template, (center, center), r, 255, 2)
    return template

def find_vortex_by_template(frame_path, template, search_right=True):
    """使用模板匹配找涡旋"""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 阈值化获取流线
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 限制搜索区域
    if search_right:
        # 右半部分，y=0 附近
        y_start = int(h * 0.3)
        y_end = int(h * 0.7)
        x_start = int(w * 0.3)
        x_end = w
    else:
        y_start = int(h * 0.3)
        y_end = int(h * 0.7)
        x_start = 0
        x_end = int(w * 0.5)
    
    search_region = binary[y_start:y_end, x_start:x_end]
    
    # 模板匹配
    result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    
    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val < 0.3:  # 匹配度太低
        return None
    
    # 转换回原图坐标
    cx = max_loc[0] + x_start + template.shape[1] // 2
    cy = max_loc[1] + y_start + template.shape[0] // 2
    
    return (cx, cy, max_val)

# 创建模板
template = create_vortex_template(150)

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
        result = find_vortex_by_template(frame_path, template, search_right=True)
        
        if result is not None:
            cx, cy, score = result
            positions.append((t_val, cx, cy, score))
            print(f"  t={t_val}: 涡旋中心 ({cx}, {cy}), 匹配度={score:.3f}")
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
