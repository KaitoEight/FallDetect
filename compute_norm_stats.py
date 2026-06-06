"""
Tính norm_mean.npy và norm_std.npy từ URFD local.
Phải chạy script này 1 lần trước khi chạy Demo_vid.py.
"""
import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import mediapipe as mp
import urllib.request

WINDOW      = 30
STEP        = WINDOW // 2   # 15, khớp notebook
INPUT_DIM   = 99            # 33 joints × 3 (x, y, z)
PROJECT_DIR = "D:/Project/falldetection_model-2"
POSE_MODEL  = os.path.join(PROJECT_DIR, "pose_landmarker_lite.task")

DATA_ROOTS = [
    os.path.join(PROJECT_DIR, "UR_fall_detection_dataset_cam0_rgb"),
    os.path.join(PROJECT_DIR, "UR_fall_detection_dataset_cam1_rgb"),
]
DATA_ROOTS = [r for r in DATA_ROOTS if os.path.exists(r)]
print(f"Datasets: {[os.path.basename(r) for r in DATA_ROOTS]}")

if not DATA_ROOTS:
    print("Khong tim thay dataset! Can co thu muc UR_fall_detection_dataset_cam0_rgb")
    exit(1)

# Tải pose model nếu chưa có
if not os.path.exists(POSE_MODEL):
    print("Dang tai pose landmarker model (~7MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        POSE_MODEL
    )

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

def extract_xyz(frame_rgb):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    res = landmarker.detect(mp_img)
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
    return np.zeros(INPUT_DIM, dtype=np.float32)

all_sequences = []
total_folders = 0

for root in DATA_ROOTS:
    for folder in sorted(os.listdir(root)):
        fpath = os.path.join(root, folder)
        if not os.path.isdir(fpath):
            continue
        files = sorted(f for f in os.listdir(fpath) if f.endswith('.png'))
        if len(files) < WINDOW:
            continue

        frame_buffer = []
        for fname in files:
            data = np.fromfile(os.path.join(fpath, fname), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                frame_buffer.append(np.zeros(INPUT_DIM, dtype=np.float32))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(extract_xyz(rgb))

        n_before = len(all_sequences)
        for i in range(0, len(frame_buffer) - WINDOW + 1, STEP):
            seq = np.stack(frame_buffer[i:i + WINDOW])  # (30, 99)
            if not np.all(seq == 0):
                all_sequences.append(seq)

        added = len(all_sequences) - n_before
        print(f"  {folder}: {len(files)} frames → {added} sequences")
        total_folders += 1

landmarker.close()
print(f"\nTong: {total_folders} thu muc, {len(all_sequences)} sequences")

if len(all_sequences) < 10:
    print("Qua it sequences! Kiem tra lai dataset.")
    exit(1)

X = np.array(all_sequences, dtype=np.float32)  # (N, 30, 99)
print(f"Shape: {X.shape}")

# Normalize theo cach cua notebook: mean/std tren toan bo frames va sequences
mean = X.mean(axis=(0, 1))         # (99,)
std  = X.std(axis=(0, 1)) + 1e-8   # (99,)

print(f"Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
print(f"Std  range: [{std.min():.3f}, {std.max():.3f}]")

np.save(os.path.join(PROJECT_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(PROJECT_DIR, "norm_std.npy"),  std)
print(f"\n[OK] Da luu norm_mean.npy va norm_std.npy")
