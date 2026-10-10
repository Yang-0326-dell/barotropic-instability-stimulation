#!/usr/bin/env python3
"""跟踪涡旋中心位置随时间变化"""
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

# 存储结果
results = {}

def find_vortex_center(frame_path, search_region='right'):
    """
    从帧图像中找到涡旋中心
    返回 (x_pixel, y_pixel) 或 None
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return None
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 图像尺寸
    h, w = gray.shape
    
    # 涡旋中心通常在 y≈0 附近（图像中部）
    # 搜索区域：右半部分或左半部分
    if search_region == 'right':
        # 右涡旋：x > w/3
        search_mask = np.zeros_like(gray)
        search_mask[:, int(w/3):] = 255
    else:
        # 左涡旋：x < w/3
        search_mask = np.zeros_like(gray)
        search_mask[:, :int(w/3)] = 255
    
    # 阈值化：黑色流线
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 应用搜索区域掩码
    binary_masked = cv2.bitwise_and(binary, binary, mask=search_mask)
    
    # 限制在 y=0 附近（中间 40% 的区域）
    y_center = h // 2
    y_band = int(h * 0.2)
    band_mask = np.zeros_like(gray)
    band_mask[y_center-y_band:y_center+y_band, :] = 255
    binary_band = cv2.bitwise_and(binary_masked, binary_masked, mask=band_mask)
    
    # 找到流线密度最高的区域（涡旋中心有大量闭合流线）
    # 使用形态学操作增强闭合区域
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(binary_band, kernel, iterations=1)
    
    # 找到连通区域
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    # 找到最大的连通区域（应该是涡旋）
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 计算质心
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx, cy)

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
        center = find_vortex_center(frame_path, 'right')
        
        if center is not None:
            cx, cy = center
            # 转换像素坐标到物理坐标
            # 图像宽度对应 Lx=10，需要知道图像的绘图区域
            # 假设绘图区域从 x_pixel_start 到 x_pixel_end 对应 0 到 Lx
            # 这里先记录像素坐标，后面再转换
            positions.append((t_val, cx, cy))
            print(f"  t={t_val}: 涡旋中心像素位置 ({cx}, {cy})")
        else:
            print(f"  t={t_val}: 未找到涡旋")
    
    results[a_val] = positions

# 保存结果
for a_val, positions in results.items():
    np.savez(results_dir / f"{a_val}_vortex_positions.npz",
             positions=np.array(positions))

print("\n结果已保存到", results_dir)
