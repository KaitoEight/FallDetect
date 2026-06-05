import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
SEQ_LEN = 20
DATA_ROOT = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb"
MODEL_PATH = "D:/Project/falldetection_model-2/pose_landmarker_lite.task"

# Tải model pose landmarker nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("Dang tai model pose landmarker (~7MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Da tai model!")

# Khởi tạo bộ dò tìm khung xương (Task API mới)
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

# Hàm bóc tách 132 con số tọa độ từ 1 khung hình
def extract_keypoints(frame_rgb):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]  # người đầu tiên
        keypoints = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
        return keypoints
    else:
        return np.zeros(33 * 4)

def load_and_preprocess_video(folder, seq_len=SEQ_LEN):
    files = sorted(os.listdir(folder))
    frames_keypoints = []

    for file in files:
        if not file.endswith(".png"): continue
        file_path = os.path.join(folder, file)

        frame_data = np.fromfile(file_path, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        keypoints = extract_keypoints(frame_rgb)
        frames_keypoints.append(keypoints)

    if len(frames_keypoints) >= seq_len:
        idx = np.linspace(0, len(frames_keypoints)-1, seq_len).astype(int)
        frames_keypoints = [frames_keypoints[i] for i in idx]
    else:
        frames_keypoints += [np.zeros(33 * 4)] * (seq_len - len(frames_keypoints))

    return np.array(frames_keypoints)

X_raw, y_raw = [], []
print("Dang chay pipeline tien xu ly du lieu...")

for folder in os.listdir(DATA_ROOT):
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path): continue

    frames = load_and_preprocess_video(folder_path)
    X_raw.append(frames)

    if "fall" in folder.lower():
        y_raw.append(1)
    else:
        y_raw.append(0)

landmarker.close()

X_raw = np.array(X_raw)
y_raw = to_categorical(np.array(y_raw), num_classes=2)

# Phân chia 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Lưu file
save_path = "D:/Project/falldetection_model-2/processed_data.npz"
np.savez_compressed(save_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
print("Da xu ly va luu du lieu thanh cong!")
print(f"Train: {X_train.shape[0]} mau | Val: {X_val.shape[0]} mau | Test: {X_test.shape[0]} mau")
