import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 64
SEQ_LEN  = 30   # khớp với notebook WINDOW=30
MODEL_PATH      = "fall_lstm_best.pt"
NORM_MEAN_PATH  = "D:/Project/falldetection_model-2/norm_mean.npy"
NORM_STD_PATH   = "D:/Project/falldetection_model-2/norm_std.npy"
POSE_MODEL_PATH = "D:/Project/falldetection_model-2/pose_landmarker_lite.task"

TEST_FOLDER = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb/fall-12-cam0-rgb"
# TEST_FOLDER = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb/adl-03-cam0-rgb"

if not os.path.exists(MODEL_PATH):
    print(f"Khong tim thay file mo hinh {MODEL_PATH}. Chay Train_model.py truoc!")
    exit()

print("Dang tai mo hinh AI...")
USE_PYTORCH = MODEL_PATH.endswith(".pt")

if USE_PYTORCH:
    import torch
    from fall_model import FallDetectNet
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        input_size = checkpoint.get('input_size', 99)
    else:
        state_dict = checkpoint
        input_size = state_dict['lstm.weight_ih_l0'].shape[1]

    pt_model = FallDetectNet(input_size=input_size).to(DEVICE)
    pt_model.load_state_dict(state_dict)
    pt_model.eval()
    model = None
    IS_KEYPOINT = True
    print(f"[PyTorch] FallDetectNet loaded | input_size={input_size} | device: {DEVICE}")

    if os.path.exists(NORM_MEAN_PATH) and os.path.exists(NORM_STD_PATH):
        norm_mean = np.load(NORM_MEAN_PATH)  # (99,)
        norm_std  = np.load(NORM_STD_PATH)   # (99,)
        print(f"[OK] Normalization: mean=[{norm_mean.min():.3f}, {norm_mean.max():.3f}]")
    else:
        norm_mean, norm_std = None, None
        print("[WARN] Khong co norm_mean.npy / norm_std.npy!")
        print("       Chay compute_norm_stats.py truoc de tao file nay.")
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    input_shape = model.input_shape
    IS_KEYPOINT = len(input_shape) == 3
    print(f"[Keras] Input: {input_shape}")

print("Tai mo hinh thanh cong!\n")

import mediapipe as mp
import urllib.request

if IS_KEYPOINT:
    if not os.path.exists(POSE_MODEL_PATH):
        print("Dang tai pose landmarker model (~7MB)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
            POSE_MODEL_PATH
        )
    _pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    pose_model = mp.tasks.vision.PoseLandmarker.create_from_options(_pose_opts)
    print("MediaPipe Pose Landmarker: OK\n")
else:
    pose_model = None

# Lưu landmarks thô để kiểm tra hướng cơ thể + track vận tốc hông
_last_raw_landmarks = None
_hip_y_history      = []   # lịch sử vị trí y của hông (15 frame gần nhất)

def extract_keypoints(frame_rgb):
    global _last_raw_landmarks, _hip_y_history
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result   = pose_model.detect(mp_image)
    if result.pose_landmarks:
        lms = result.pose_landmarks[0]
        _last_raw_landmarks = lms
        # Cập nhật lịch sử hip_y
        hip_y = (lms[23].y + lms[24].y) / 2
        _hip_y_history.append(hip_y)
        if len(_hip_y_history) > 15:
            _hip_y_history.pop(0)
        if USE_PYTORCH and input_size == 99:
            return np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
        else:
            return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=np.float32).flatten()
    _last_raw_landmarks = None
    return np.zeros(input_size if USE_PYTORCH else 132, dtype=np.float32)


def is_hip_dropped_fast(hip_hist, window=12, threshold=0.12):
    """
    True nếu hông tụt xuống nhanh trong `window` frame gần nhất.
    Té ngã: hông tụt mạnh (Δy > 0.12 trong 12 frame).
    Ngồi xuống: hông tụt chậm (Δy nhỏ).
    """
    if len(hip_hist) < window:
        return False
    return (hip_hist[-1] - hip_hist[-window]) > threshold


def is_body_flat(landmarks):
    """
    True nếu cơ thể đang nằm ngang.
    Yêu cầu CẢ HAI: vai gần hông VÀ bounding box nằm ngang.
    """
    if landmarks is None:
        return False

    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_y      = (l_hip.y + r_hip.y) / 2

    # Điều kiện 1: vai ngang với hông (thân nằm phẳng)
    # Ngồi: shoulder_y cách hip_y ~0.2-0.3 → không thỏa
    # Nằm : shoulder_y ≈ hip_y < 0.10 → thỏa
    torso_flat = abs(shoulder_y - hip_y) < 0.10

    # Điều kiện 2: bounding box cơ thể rộng hơn cao (nằm ngang)
    all_y = [lm.y for lm in landmarks]
    all_x = [lm.x for lm in landmarks]
    body_h = max(all_y) - min(all_y)
    body_w = max(all_x) - min(all_x)
    body_wide = body_h < body_w * 0.9   # chiều cao < chiều rộng → nằm ngang

    return torso_flat or body_wide


if not os.path.exists(TEST_FOLDER):
    print(f"Khong tim thay thu muc: {TEST_FOLDER}")
    exit()

# Tham số lọc
FALL_THRESHOLD = 0.70    # ngưỡng xác suất fall của model
CONFIRM_FRAMES = 4       # cần N frame liên tiếp để cảnh báo

print("BAT DAU PHAT VIDEO... (Nhan phim 'q' de thoat)")
if norm_mean is None:
    print("[WARN] Chay khong co normalization — ket qua co the sai!")

files = sorted(os.listdir(TEST_FOLDER))
frame_buffer = []
fall_streak  = 0
alert_active = False

for file in files:
    if not file.endswith(".png"): continue
    file_path = os.path.join(TEST_FOLDER, file)

    frame_data = np.fromfile(file_path, dtype=np.uint8)
    orig_frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    display_frame = cv2.resize(orig_frame, (640, 480))

    frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    if IS_KEYPOINT:
        feature = extract_keypoints(frame_rgb)
    else:
        feature = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0

    frame_buffer.append(feature)
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)

    label_text = "Dang cho du lieu... ({}/{})".format(len(frame_buffer), SEQ_LEN)
    text_color = (0, 255, 255)

    if len(frame_buffer) == SEQ_LEN:
        if USE_PYTORCH:
            kp_arr = np.array(frame_buffer, dtype=np.float32)  # (30, 99)
            if input_size == 264:
                velocity = np.zeros_like(kp_arr)
                velocity[1:] = kp_arr[1:] - kp_arr[:-1]
                input_arr = np.concatenate([kp_arr, velocity], axis=1)
            else:
                input_arr = kp_arr

            if norm_mean is not None:
                input_arr = (input_arr - norm_mean) / norm_std

            with torch.no_grad():
                t     = torch.tensor(input_arr[None]).to(DEVICE)
                probs = torch.softmax(pt_model(t), dim=1).cpu().numpy()[0]
            prob_not_fall = probs[0] * 100
            prob_fall     = probs[1] * 100
        else:
            kp_arr = np.array(frame_buffer)
            if IS_KEYPOINT:
                velocity = np.zeros_like(kp_arr)
                velocity[1:] = kp_arr[1:] - kp_arr[:-1]
                input_arr = np.concatenate([kp_arr, velocity], axis=1)
            else:
                input_arr = kp_arr
            pred          = model.predict(np.expand_dims(input_arr, 0), verbose=0)
            prob_not_fall = pred[0][0] * 100
            prob_fall     = pred[0][1] * 100

        # Kiểm tra 3 tầng: model + vận tốc + hướng cơ thể
        model_says_fall = prob_fall >= FALL_THRESHOLD * 100
        hip_fast        = is_hip_dropped_fast(_hip_y_history)
        body_flat       = is_body_flat(_last_raw_landmarks)

        # Xác nhận ngã: model tin AND (hông tụt nhanh OR cơ thể đã nằm)
        confirmed_fall = model_says_fall and (hip_fast or body_flat)

        if confirmed_fall:
            fall_streak += 1
        else:
            fall_streak  = 0
            alert_active = False

        if fall_streak >= CONFIRM_FRAMES:
            alert_active = True

        # Hiển thị thông tin debug
        flags = []
        if model_says_fall: flags.append(f"ML:{prob_fall:.0f}%")
        if hip_fast:        flags.append("VEL")
        if body_flat:       flags.append("FLAT")
        flag_str = "|".join(flags) if flags else "normal"

        if alert_active:
            label_text = f"!!! TE NGA !!! ({prob_fall:.1f}%)"
            text_color = (0, 0, 255)
            cv2.rectangle(display_frame, (0, 0), (640, 480), (0, 0, 255), 10)
        else:
            label_text = f"Binh Thuong | {flag_str} | streak={fall_streak}"
            text_color = (0, 255, 0)

    cv2.putText(display_frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
    cv2.imshow("He Thong Giam Sat Te Nga", display_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

if pose_model and USE_PYTORCH:
    pose_model.close()
cv2.destroyAllWindows()
