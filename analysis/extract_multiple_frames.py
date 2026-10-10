#!/usr/bin/env python3
"""提取多个时间步的帧并跟踪涡旋中心"""
import cv2
import numpy as np
from pathlib import Path

# 视频路径
video_dir = Path("/Users/coalminer/Downloads")
output_dir = Path("/Users/coalminer/Desktop/流体模拟/HW1/vortex_tracking")
output_dir.mkdir(exist_ok=True)

# 视频配置
videos = {
    "a0.05": "solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_pert.mp4",
    "a0.08": "solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_pert.mp4",
    "a0.1": "solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4",
    "a0.2": "solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_pert.mp4",
}

# 物理参数
Lx = 10
T = 50
dt = 0.1
total_frames = 501  # 0 to 50.0, step 0.1

# 提取的时间步
time_steps = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
frame_indices = [int(t / dt) for t in time_steps]

for a_val, video_name in videos.items():
    video_path = video_dir / video_name
    print(f"\n处理 {a_val}: {video_path.name}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  无法打开视频")
        continue
    
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  视频总帧数: {total_video_frames}")
    
    # 提取指定时间步的帧
    frames_dir = output_dir / a_val
    frames_dir.mkdir(exist_ok=True)
    
    extracted_frames = {}
    for t_val, frame_idx in zip(time_steps, frame_indices):
        if frame_idx >= total_video_frames:
            print(f"  t={t_val:.1f} (frame {frame_idx}) 超出范围")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # 保存帧
            frame_path = frames_dir / f"t{t_val:.0f}.png"
            cv2.imwrite(str(frame_path), frame)
            extracted_frames[t_val] = frame
            print(f"  保存 t={t_val:.1f} -> {frame_path.name}")
    
    cap.release()
    print(f"  共提取 {len(extracted_frames)} 帧")

print("\n完成帧提取")
