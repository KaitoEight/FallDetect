import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from sklearn.model_selection import train_test_split

# ── Tham số — khớp với notebook Kaggle ──────────────────────────────────────
WINDOW      = 30          # số frame mỗi sequence (notebook dùng WINDOW=30)
STEP        = WINDOW // 2 # bước trượt = 15
INPUT_DIM   = 99          # 33 joints × 3 (x, y, z) — không dùng visibility
PROJECT_DIR = "D:/Project/falldetection_model-2"
POSE_MODEL  = os.path.join(PROJECT_DIR, "pose_landmarker_lite.task")

DATA_ROOTS = [
    os.path.join(PROJECT_DIR, "UR_fall_detection_dataset_cam0_rgb"),
    os.path.join(PROJECT_DIR, "UR_fall_detection_dataset_cam1_rgb"),
    os.path.join(PROJECT_DIR, "fall_dataset"),
]
DATA_ROOTS = [r for r in DATA_ROOTS if os.path.exists(r)]
print(f"Su dung {len(DATA_ROOTS)} dataset(s): {[os.path.basename(r) for r in DATA_ROOTS]}")

# ── Pose landmarker ───────────────────────────────────────────────────────────
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
    """Trả về vector (99,) gồm x,y,z của 33 landmarks."""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    res    = landmarker.detect(mp_img)
    if res.pose_landmarks:
        lms = res.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
    return np.zeros(INPUT_DIM, dtype=np.float32)

def load_folder_sequences(folder_path):
    """
    Đọc tất cả frame .png trong folder, trích keypoints,
    rồi tạo sliding-window sequences (WINDOW, 99).
    """
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".png"))
    if len(files) < WINDOW:
        return []

    frames = []
    for fname in files:
        data  = np.fromfile(os.path.join(folder_path, fname), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            frames.append(np.zeros(INPUT_DIM, dtype=np.float32))
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(extract_xyz(rgb))

    seqs = []
    for i in range(0, len(frames) - WINDOW + 1, STEP):
        seq = np.stack(frames[i:i + WINDOW])  # (30, 99)
        if not np.all(seq == 0):              # bỏ sequence toàn 0
            seqs.append(seq.astype(np.float32))
    return seqs

def augment(seq, n_noise=4):
    """
    Tăng cường 1 sequence: noise + lật ngang.
    Trả về list các sequence mới (không kèm original).
    seq: (30, 99) — x,y,z lặp 33 lần
    """
    results = []
    # Gaussian noise
    for _ in range(n_noise):
        noisy = np.clip(seq + np.random.normal(0, 0.01, seq.shape), 0, 1)
        results.append(noisy.astype(np.float32))

    # Lật ngang: đảo tọa độ x (index 0, 3, 6, ... — mỗi joint 3 phần tử x,y,z)
    flipped = seq.copy()
    flipped[:, 0::3] = 1.0 - flipped[:, 0::3]  # x = 1 - x
    results.append(flipped.astype(np.float32))

    return results


# ── Trích xuất toàn bộ data ───────────────────────────────────────────────────
X_raw, y_raw = [], []
print("\nDang xu ly du lieu...")

for root in DATA_ROOTS:
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        label = 1 if "fall" in folder.lower() else 0
        seqs  = load_folder_sequences(folder_path)
        if seqs:
            X_raw.extend(seqs)
            y_raw.extend([label] * len(seqs))
            print(f"  {folder}: {len(seqs)} seq (label={label})")

landmarker.close()

X_raw = np.array(X_raw, dtype=np.float32)  # (N, 30, 99)
y_raw = np.array(y_raw, dtype=np.int64)

counts = np.bincount(y_raw)
print(f"\nTong: {len(X_raw)} sequences")
print(f"  Fall  (1): {counts[1]}  ({counts[1]/len(X_raw)*100:.1f}%)")
print(f"  ADL   (0): {counts[0]}  ({counts[0]/len(X_raw)*100:.1f}%)")

if len(X_raw) < 50:
    print("Qua it sequences! Kiem tra lai DATA_ROOTS va dataset.")
    exit(1)

# ── Normalize toàn bộ TRƯỚC khi split (khớp notebook) ───────────────────────
mean = X_raw.mean(axis=(0, 1))         # (99,)
std  = X_raw.std(axis=(0, 1)) + 1e-8   # (99,)
X_norm = (X_raw - mean) / std

np.save(os.path.join(PROJECT_DIR, "norm_mean.npy"), mean)
np.save(os.path.join(PROJECT_DIR, "norm_std.npy"),  std)
print(f"\nNorm stats: mean=[{mean.min():.3f}, {mean.max():.3f}] | std=[{std.min():.3f}, {std.max():.3f}]")
print("Da luu norm_mean.npy va norm_std.npy")

# ── Split 80 / 10 / 10 ───────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y_raw, test_size=0.2, random_state=42, stratify=y_raw)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ── Augment chỉ trên train (sau split — không data leakage) ─────────────────
aug_X, aug_y = list(X_train), list(y_train)
for seq, lbl in zip(X_train, y_train):
    for s in augment(seq):
        aug_X.append(s)
        aug_y.append(lbl)

X_train = np.array(aug_X, dtype=np.float32)
y_train = np.array(aug_y, dtype=np.int64)

print(f"\nSau augment:")
print(f"  Train : {X_train.shape} | fall%: {y_train.mean()*100:.1f}%")
print(f"  Val   : {X_val.shape}   | fall%: {y_val.mean()*100:.1f}%")
print(f"  Test  : {X_test.shape}  | fall%: {y_test.mean()*100:.1f}%")

save_path = os.path.join(PROJECT_DIR, "processed_data.npz")
np.savez_compressed(save_path,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val,     y_val=y_val,
                    X_test=X_test,   y_test=y_test)
print(f"\nDa luu {save_path}")
