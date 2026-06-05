import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt


data_path = "D:/Project/falldetection_model-2/processed_data.npz"
print("Đang nạp dữ liệu từ ổ cứng lên RAM...")

data = np.load(data_path)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

print("✅ Đã nạp dữ liệu thành công!")
print(f"Số lượng video tập Train: {X_train.shape[0]}")
print(f"Số lượng video tập Validation: {X_val.shape[0]}")
print(f"Số lượng video tập Test: {X_test.shape[0]}")

SEQ_LEN = X_train.shape[1]  # 20 frames
IS_KEYPOINT = X_train.ndim == 3  # True nếu data keypoints (20, 132), False nếu ảnh (20, 64, 64, 3)

print("\n--- KHỞI TẠO MÔ HÌNH ---")
model = Sequential()

if IS_KEYPOINT:
    # Mô hình LSTM cho dữ liệu keypoints shape (SEQ_LEN, 132)
    NUM_FEATURES = X_train.shape[2]
    print(f"Phát hiện dữ liệu KEYPOINTS ({SEQ_LEN} frames x {NUM_FEATURES} features) -> dùng LSTM")
    model.add(LSTM(128, return_sequences=True, input_shape=(SEQ_LEN, NUM_FEATURES)))
    model.add(Dropout(0.3))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.3))
else:
    # Mô hình CNN+LSTM cho dữ liệu ảnh shape (SEQ_LEN, 64, 64, 3)
    IMG_SIZE = X_train.shape[2]
    print(f"Phát hiện dữ liệu ẢNH ({SEQ_LEN} frames x {IMG_SIZE}x{IMG_SIZE}) -> dùng CNN+LSTM")
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'), input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# hàm Callbacks giám sát quá trình huấn luyện
callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_fall_detection_model.h5', monitor='val_loss', save_best_only=True, verbose=1)]

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
history = model.fit(X_train, y_train,validation_data=(X_val, y_val),epochs=50,batch_size=8,callbacks=callbacks)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=========================================")
print(f"=> ĐỘ CHÍNH XÁC TRÊN TẬP TEST: {test_acc*100:.2f}%")
print(f"=========================================")

# Vẽ biểu đồ 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Biểu đồ Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
axes[0].set_title('Biểu đồ độ chính xác (Accuracy)')
axes[0].set_xlabel('Vòng huấn luyện (Epoch)')
axes[0].set_ylabel('Độ chính xác')
axes[0].legend()

# Biểu đồ Loss
axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
axes[1].plot(history.history['val_loss'], label='Validation Loss', color='orange')
axes[1].set_title('Biểu đồ hàm mất mát (Loss)')
axes[1].set_xlabel('Vòng huấn luyện (Epoch)')
axes[1].set_ylabel('Sai số')
axes[1].legend()

plt.tight_layout()
plt.show()