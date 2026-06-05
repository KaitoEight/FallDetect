import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt # Thêm thư viện vẽ đồ thị

IMG_SIZE = 64
SEQ_LEN = 20

# Tải mô hình đã huấn luyện
model_path = "best_fall_detection_model.h5"
if not os.path.exists(model_path):
    print(f"Không tìm thấy file mô hình {model_path}. Bạn cần chạy file huấn luyện trước!")
    exit()

print("Đang tải bộ não AI tốt nhất...")
model = tf.keras.models.load_model(model_path)
print("✅ Tải mô hình thành công!")

# Hàm xử lí chuỗi 
def prepare_video_sequence(folder_path, seq_len=SEQ_LEN, img_size=IMG_SIZE):
    files = sorted(os.listdir(folder_path))
    frames = []
    for file in files:
        if not file.endswith(".png"): continue
        file_path = os.path.join(folder_path, file)
        
        frame_data = np.fromfile(file_path, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    if len(frames) >= seq_len:
        idx = np.linspace(0, len(frames)-1, seq_len).astype(int)
        frames = [frames[i] for i in idx]
    else:
        frames += [np.zeros((img_size, img_size, 3))] * (seq_len - len(frames))
    
    # Thêm 1 chiều ở đầu (Batch size = 1) để khớp định dạng đầu vào của mô hình
    return np.expand_dims(np.array(frames), axis=0)


# Chọn 1 video test thử
TEST_FOLDER = "D:/Project/falldetection_model-2/UR_fall_detection_dataset_cam0_rgb/fall-12-cam0-rgb"
# adl-04-cam0-rgb
# adl-01-cam0-rgb

if os.path.exists(TEST_FOLDER):
    print(f"\nĐang đọc video từ thư mục: {os.path.basename(TEST_FOLDER)}")
    input_data = prepare_video_sequence(TEST_FOLDER)
    
    prediction = model.predict(input_data, verbose=0)
    
    # prediction trả về dạng [[Xác_suất_Không_Ngã, Xác_suất_Ngã]]
    prob_not_fall = prediction[0][0] * 100
    prob_fall = prediction[0][1] * 100
    
    print("\n" + "="*40)
    print(" KẾT QUẢ CHẨN ĐOÁN CỦA AI:")
    print(f" - Bình thường (Not Fall): {prob_not_fall:.2f}%")
    print(f" - Té ngã (Fall): {prob_fall:.2f}%")
    print("="*40)
    
    # Xác định trạng thái để in và vẽ
    is_falling = prob_fall > prob_not_fall
    
    if is_falling:
        print("🚨 CẢNH BÁO: Phát hiện có hành động TÉ NGÃ trong video!")
    else:
        print("✅ AN TOÀN: Người trong video hoạt động bình thường.")
        
    # ==========================================
    # KHỐI LỆNH VISUALIZE HIỂN THỊ TRỰC QUAN
    # ==========================================
    print("\nĐang khởi tạo giao diện trực quan...")
    
    # Tạo một khung hình gồm 1 hàng, 6 cột (5 cột cho ảnh, 1 cột cho biểu đồ)
    fig, axes = plt.subplots(1, 6, figsize=(16, 4))
    
    # Lấy chuỗi 20 khung hình từ dữ liệu đầu vào (bỏ chiều batch_size)
    sequence_frames = input_data[0] 
    
    # Chọn ra 5 khung hình đại diện (đầu, giữa, cuối) để hiển thị
    display_indices = [0, 4, 9, 14, 19]
    
    for i, idx in enumerate(display_indices):
        axes[i].imshow(sequence_frames[idx])
        axes[i].set_title(f"Frame {idx+1}")
        axes[i].axis("off")
        
    # Cột cuối cùng (thứ 6): Vẽ biểu đồ cột hiển thị xác suất
    labels = ['Bình thường', 'Té ngã']
    probs = [prob_not_fall, prob_fall]
    
    # Đổi màu đỏ nếu ngã, xanh nếu an toàn
    bar_colors = ['#2ecc71' if not is_falling else '#95a5a6', 
                  '#e74c3c' if is_falling else '#95a5a6']
                  
    axes[5].bar(labels, probs, color=bar_colors)
    axes[5].set_ylim(0, 100)
    axes[5].set_ylabel('Xác suất (%)')
    axes[5].set_title('Mức độ tin cậy của AI', fontweight='bold')
    
    # Đặt tiêu đề lớn cho toàn bộ cửa sổ
    status_text = "🚨 PHÁT HIỆN TÉ NGÃ" if is_falling else "✅ AN TOÀN"
    title_color = "red" if is_falling else "green"
    fig.suptitle(f"Kết quả phân tích chuỗi video: {os.path.basename(TEST_FOLDER)} | Trạng thái: {status_text}", 
                 fontsize=16, fontweight='bold', color=title_color)
                 
    plt.tight_layout()
    plt.show()

else:
    print(f"Không tìm thấy thư mục test: {TEST_FOLDER}")