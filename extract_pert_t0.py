import cv2
import os

video_path = '/Users/coalminer/Downloads/solve_2D_nonlinear_case1_ap0.1_k1_pert.mp4'
output_dir = '/Users/coalminer/Downloads/frames_dt1'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Extract frame at t=0
frame_num = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
ret, frame = cap.read()
if ret:
    output_path = os.path.join(output_dir, f'solve_2D_nonlinear_case1_ap0.1_k1_pert_t00.png')
    cv2.imwrite(output_path, frame)
    print(f"Saved frame at t=0 (frame {frame_num})")

cap.release()
