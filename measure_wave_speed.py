import cv2
import numpy as np
import os
import glob

DL = os.path.expanduser("~/Downloads")
OUT = "/Users/coalminer/Desktop/流体模拟/HW1/wave_speed"
os.makedirs(OUT, exist_ok=True)

VIDEOS = {
    "a0.05": "solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert.mp4",
    "a0.08": "solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert.mp4",
    "a0.1": "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4",
    "a0.2": "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4",
}

# 视频参数（从 solve_k1.py 等脚本）
Lx = 10.0
T_total = 50.0
N_frames = 501
dt = T_total / (N_frames - 1)  # 0.1

# 绘图区域校准（从视频帧目测）
# x 轴：像素 100..2800 对应 [0, 10]
# y 轴：像素 100..1100 对应 [-2, 2]
X_PIX_MIN, X_PIX_MAX = 100, 2800
Y_PIX_MIN, Y_PIX_MAX = 100, 1100
X_DATA_MIN, X_DATA_MAX = 0.0, 10.0
Y_DATA_MIN, Y_DATA_MAX = -2.0, 2.0

def pix_to_x(px):
    return X_DATA_MIN + (px - X_PIX_MIN) / (X_PIX_MAX - X_PIX_MIN) * (X_DATA_MAX - X_DATA_MIN)

def pix_to_y(py):
    return Y_DATA_MAX - (py - Y_PIX_MIN) / (Y_PIX_MAX - Y_PIX_MIN) * (Y_DATA_MAX - Y_DATA_MIN)


def extract_wavepacket_center(frame_bgr):
    """从渲染帧提取右行波包中心 x 坐标（物理单位）"""
    h, w = frame_bgr.shape[:2]
    # 裁剪绘图区域
    crop = frame_bgr[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]
    # 转 HSV 提取蓝色（负扰动）和红色（正扰动）
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # 蓝色：H ~ 100-140
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (140, 255, 255))
    # 红色：H ~ 0-10 或 160-180
    red_mask1 = cv2.inRange(hsv, (0, 80, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 80, 80), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    # 扰动 = 蓝色 ∪ 红色
    pert_mask = cv2.bitwise_or(blue_mask, red_mask)
    # 取 y=0 附近 ±50 像素的条带（物理 y ∈ [-0.5, 0.5]）
    y_center_pix = (Y_PIX_MAX - Y_PIX_MIN) // 2
    band = pert_mask[y_center_pix-50:y_center_pix+50, :]
    # 沿 y 积分得到扰动强度剖面 I(x_pix)
    profile = band.sum(axis=0).astype(float)
    # 找右行波包：x_pix > 初始位置（初始波包在 x≈1，对应像素 ~380）
    # 取 x_pix > 500 的部分
    profile_right = profile.copy()
    profile_right[:500] = 0
    if profile_right.sum() < 100:
        return None  # 无扰动
    # 质心
    x_pix = np.arange(len(profile_right))
    x_center_pix = (x_pix * profile_right).sum() / profile_right.sum()
    return pix_to_x(x_center_pix)


results = {}
for label, fname in VIDEOS.items():
    path = os.path.join(DL, fname)
    cap = cv2.VideoCapture(path)
    centers = []
    times = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx * dt
        xc = extract_wavepacket_center(frame)
        if xc is not None:
            centers.append(xc)
            times.append(t)
        frame_idx += 1
    cap.release()
    results[label] = (np.array(times), np.array(centers))
    print(f"{label}: {len(centers)} frames, x range [{min(centers):.2f}, {max(centers):.2f}]")

# 拟合速度：x(t) = x0 + v*t
print("\n=== 传播速率 ===")
for label, (times, centers) in results.items():
    # 取 t > 5 的数据（避开初始 transient）
    mask = times > 5.0
    if mask.sum() < 10:
        print(f"{label}: 数据不足")
        continue
    t_fit = times[mask]
    x_fit = centers[mask]
    # 线性拟合
    coeffs = np.polyfit(t_fit, x_fit, 1)
    v = coeffs[0]  # 速度（物理单位 / 时间单位）
    x0 = coeffs[1]
    # 残差
    x_pred = np.polyval(coeffs, t_fit)
    rms = np.sqrt(((x_fit - x_pred)**2).mean())
    print(f"{label}: v = {v:.4f} (Lx/T), x0 = {x0:.3f}, RMS = {rms:.3f}")

# 保存数据
for label, (times, centers) in results.items():
    np.savez(os.path.join(OUT, f"{label}_wavepacket.npz"), t=times, x=centers)
print(f"\n数据已保存到 {OUT}")
