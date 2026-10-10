import cv2
import os

DL = os.path.expanduser("~/Downloads")
OUT = "/Users/coalminer/Desktop/流体模拟/HW1/last_frames"
os.makedirs(OUT, exist_ok=True)

videos = {
    "a0.05": ("solve_2D_nonlinear_case1_abg0.05_ap0.05_k1_{}.mp4"),
    "a0.08": ("solve_2D_nonlinear_case1_abg0.08_ap0.08_k1_{}.mp4"),
    "a0.1": ("solve_2D_nonlinear_case1_ap0.1_k1_{}.mp4"),
    "a0.2": ("solve_2D_nonlinear_case1_abg0.2_ap0.2_k1_{}.mp4"),
}

for a, pat in videos.items():
    for kind in ("total", "pert"):
        src = os.path.join(DL, pat.format(kind))
        cap = cv2.VideoCapture(src)
        last = None
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            n += 1
        cap.release()
        dst = os.path.join(OUT, f"{a}_{kind}.png")
        cv2.imwrite(dst, last)
        print(f"{a} {kind}: frames={n} shape={last.shape} -> {dst}")
