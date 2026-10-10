import cv2
import os

video_path = '/Users/coalminer/Downloads/solve_2D_nonlinear_case1_ap0.1_k1_total.mp4'
output_dir = '/Users/coalminer/Downloads/frames_dt1_total'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Total frames: {total_frames}")

# Extract frames at t=0, 10, 20, 50 (10 FPS, so frame numbers are 0, 100, 200, 500)
target_times = [0, 10, 20, 50]
for t in target_times:
    frame_num = int(t * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        output_path = os.path.join(output_dir, f'solve_2D_nonlinear_case1_ap0.1_k1_total_t{t:02d}.png')
        cv2.imwrite(output_path, frame)
        print(f"Saved frame at t={t} (frame {frame_num})")

cap.release()
