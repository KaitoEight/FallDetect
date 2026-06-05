import os
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
import tensorflow as tf
import urllib.request

IMG_SIZE = 64
SEQ_LEN = 20
MODEL_PATH = "best_fall_detection_model.h5"
POSE_MODEL_PATH = "D:/Project/falldetection_model-2/pose_landmarker_lite.task"

# TEST_FOLDER = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb/fall-12-cam0-rgb"
TEST_FOLDER = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb/fall-30-cam0-rgb"
# adl-04-cam0-rgb

if not os.path.exists(MODEL_PATH):
    print(f"Khong tim thay file mo hinh {MODEL_PATH}. Chay Train_model.py truoc!")
    exit()

print("Dang tai mo hinh AI...")
model = tf.keras.models.load_model(MODEL_PATH)

# Detect loai model dua vao input shape
input_shape = model.input_shape  # (None, 20, 132) hoac (None, 20, 64, 64, 3)
IS_KEYPOINT = len(input_shape) == 3
print(f"Mo hinh: {'LSTM Keypoint' if IS_KEYPOINT else 'CNN+LSTM Anh'} | Input: {input_shape}")
print("Tai mo hinh thanh cong!\n")

# Khởi tạo mediapipe nếu model dùng keypoints
landmarker = None
if IS_KEYPOINT:
    import mediapipe as mp
    if not os.path.exists(POSE_MODEL_PATH):
        print("Dang tai pose landmarker model (~7MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        urllib.request.urlretrieve(url, POSE_MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.IMAGE
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    print("MediaPipe Pose Landmarker: OK\n")

def extract_keypoints(frame_rgb):
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)
    if result.pose_landmarks:
        lms = result.pose_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms]).flatten()
    return np.zeros(33 * 4)

if not os.path.exists(TEST_FOLDER):
    print(f"Khong tim thay thu muc: {TEST_FOLDER}")
    exit()

print("BAT DAU PHAT VIDEO... (Nhan phim 'q' de thoat)")

files = sorted(os.listdir(TEST_FOLDER))
frame_buffer = []

for file in files:
    if not file.endswith(".png"): continue
    file_path = os.path.join(TEST_FOLDER, file)

    frame_data = np.fromfile(file_path, dtype=np.uint8)
    orig_frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    display_frame = cv2.resize(orig_frame, (640, 480))

    # Chuẩn bị feature cho buffer tùy theo loại model
    frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
    if IS_KEYPOINT:
        feature = extract_keypoints(frame_rgb)
    else:
        frame_resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE))
        feature = frame_resized.astype("float32") / 255.0

    frame_buffer.append(feature)
    if len(frame_buffer) > SEQ_LEN:
        frame_buffer.pop(0)

    label_text = "Dang cho du lieu... ({}/{})".format(len(frame_buffer), SEQ_LEN)
    text_color = (0, 255, 255)

    if len(frame_buffer) == SEQ_LEN:
        input_data = np.expand_dims(np.array(frame_buffer), axis=0)
        prediction = model.predict(input_data, verbose=0)
        prob_not_fall = prediction[0][0] * 100
        prob_fall = prediction[0][1] * 100

        if prob_fall > prob_not_fall:
            label_text = f"CANH BAO: TE NGA! ({prob_fall:.1f}%)"
            text_color = (0, 0, 255)
            cv2.rectangle(display_frame, (0, 0), (640, 480), (0, 0, 255), 10)
        else:
            label_text = f"Binh Thuong ({prob_not_fall:.1f}%)"
            text_color = (0, 255, 0)

    cv2.putText(display_frame, label_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
    cv2.imshow("He Thong Giam Sat Te Nga (DeepFall-Net)", display_frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

if landmarker:
    landmarker.close()
cv2.destroyAllWindows()
