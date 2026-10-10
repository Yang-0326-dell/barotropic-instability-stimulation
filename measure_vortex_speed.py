import cv2
import numpy as np
import os

DL = os.path.expanduser("~/Downloads")
OUT = "/Users/coalminer/Desktop/流体模拟/HW1/wave_speed"

VIDEOS = {
    "a0.05": "solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert.mp4",
    "a0.08": "solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert.mp4",
    "a0.1": "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4",
    "a0.2": "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4",
}

Lx = 10.0
T_total = 50.0
N_frames = 501
dt = T_total / (N_frames - 1)

X_PIX_MIN, X_PIX_MAX = 100, 2800
Y_PIX_MIN, Y_PIX_MAX = 100, 1100

def pix_to_x(px):
    return (px - X_PIX_MIN) / (X_PIX_MAX - X_PIX_MIN) * Lx

def find_vortex_peak(frame_bgr, x_min_pix=1200):
    """找右涡旋峰值位置（蓝色密度最大的 x）"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (140, 255, 255))
    h, w = blue_mask.shape
    crop = np.zeros_like(blue_mask)
    crop[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX] = blue_mask[Y_PIX_MIN:Y_PIX_MAX, X_PIX_MIN:X_PIX_MAX]
    # y=0 附近 ±80 像素
    y_center = (Y_PIX_MIN + Y_PIX_MAX) // 2
    band = crop[y_center-80:y_center+80, :]
    # 沿 y 积分得到蓝色密度剖面
    profile = band.sum(axis=0).astype(float)
    # 平滑
    kernel = np.ones(50) / 50
    profile_smooth = np.convolve(profile, kernel, mode='same')
    # 只取 x > x_min_pix
    profile_smooth[:x_min_pix] = 0
    if profile_smooth.max() < 100:
        return None
    # 找峰值
    x_peak_pix = np.argmax(profile_smooth)
    return pix_to_x(x_peak_pix), profile_smooth[x_peak_pix]

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
        if frame_idx >= 50:
            result = find_vortex_peak(frame, x_min_pix=1200)
            if result is not None:
                xc, strength = result
                centers.append(xc)
                times.append(t)
        frame_idx += 1
    cap.release()
    results[label] = (np.array(times), np.array(centers))
    print(f"{label}: {len(centers)} frames, x range [{min(centers):.2f}, {max(centers):.2f}]")

print("\n=== 右涡旋传播速率（峰值追踪）===")
for label, (times, centers) in results.items():
    # 取 t ∈ [10, 30]
    mask = (times >= 10) & (times <= 30)
    if mask.sum() < 10:
        print(f"{label}: 数据不足")
        continue
    t_fit = times[mask]
    x_fit = centers[mask]
    coeffs = np.polyfit(t_fit, x_fit, 1)
    v = coeffs[0]
    x0 = coeffs[1]
    x_pred = np.polyval(coeffs, t_fit)
    rms = np.sqrt(((x_fit - x_pred)**2).mean())
    print(f"{label}: v = {v:.4f} (Lx/T), x0 = {x0:.3f}, RMS = {rms:.3f}")
    # 打印几个数据点
    def fmt(v):
        return f"{v:.2f}" if v is not None else "N/A"
    x10 = x_fit[t_fit==10][0] if (t_fit==10).any() else None
    x20 = x_fit[t_fit==20][0] if (t_fit==20).any() else None
    x30 = x_fit[t_fit==30][0] if (t_fit==30).any() else None
    print(f"  样本：t=10 → x={fmt(x10)}, t=20 → x={fmt(x20)}, t=30 → x={fmt(x30)}")

for label, (times, centers) in results.items():
    np.savez(os.path.join(OUT, f"{label}_vortex_peak.npz"), t=times, x=centers)
print(f"\n数据已保存到 {OUT}")
