import numpy as np
import matplotlib.pyplot as plt

data_path = "D:/Project/falldetection_model-2/processed_data.npz"
print("Dang nap du lieu...")

data    = np.load(data_path)
X_train = data['X_train'].astype("float32")
y_train = data['y_train']
X_val   = data['X_val'].astype("float32")
y_val   = data['y_val']
X_test  = data['X_test'].astype("float32")
y_test  = data['y_test']

# y có thể là one-hot hoặc integer
if y_train.ndim == 2:
    y_train = np.argmax(y_train, axis=1)
    y_val   = np.argmax(y_val,   axis=1)
    y_test  = np.argmax(y_test,  axis=1)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
print(f"Feature shape: {X_train.shape[1:]}")

IS_KEYPOINT = X_train.ndim == 3
USE_PYTORCH = IS_KEYPOINT

if USE_PYTORCH:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
    from fall_model import FallDetectNet

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS     = 60
    BATCH_SIZE = 64
    LR         = 1e-3
    PATIENCE   = 12
    SAVE_PATH  = "fall_lstm_best.pt"

    print(f"\n--- PYTORCH | device: {DEVICE} ---")

    # WeightedRandomSampler để cân bằng class imbalance
    class_counts    = np.bincount(y_train)
    weights_per_cls = 1.0 / class_counts
    sample_weights  = weights_per_cls[y_train]
    sampler         = WeightedRandomSampler(sample_weights, len(sample_weights))

    def make_loader(X, y, sampler=None):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler,
                          shuffle=(sampler is None))

    train_loader = make_loader(X_train, y_train, sampler=sampler)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    input_size = X_train.shape[2]  # 99
    model      = FallDetectNet(input_size=input_size).to(DEVICE)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Weighted loss: phạt nặng hơn khi bỏ sót fall
    fall_weight = float(class_counts[0]) / class_counts[1]
    print(f"Fall class weight: {fall_weight:.2f}x")
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1.0, fall_weight]).to(DEVICE)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=6, factor=0.5)

    best_val_acc = 0.0
    patience_cnt = 0
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("--- BAT DAU HUAN LUYEN ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss, t_correct, t_total = 0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * len(yb)
            t_correct += (out.argmax(1) == yb).sum().item()
            t_total   += len(yb)

        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out    = model(xb)
                loss   = criterion(out, yb)
                v_loss    += loss.item() * len(yb)
                v_correct += (out.argmax(1) == yb).sum().item()
                v_total   += len(yb)

        tl = t_loss / t_total;  ta = t_correct / t_total
        vl = v_loss / v_total;  va = v_correct / v_total
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        scheduler.step(va)

        print(f"Epoch {epoch:3d}/{EPOCHS} | loss={tl:.4f} acc={ta:.3f} | val_loss={vl:.4f} val_acc={va:.3f}")

        if va > best_val_acc:
            best_val_acc = va
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  => Luu model tot nhat: val_acc={va:.4f}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping tai epoch {epoch}")
                break

    # Test
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb  = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)

    print(f"\n=========================================")
    print(f"=> DO CHINH XAC TREN TAP TEST: {correct/total*100:.2f}%")
    print(f"=========================================")

else:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (TimeDistributed, Conv2D, MaxPooling2D,
                                         GlobalAveragePooling2D, LSTM, Dense, Dropout)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical

    y_train = to_categorical(y_train, 2)
    y_val   = to_categorical(y_val,   2)
    y_test  = to_categorical(y_test,  2)

    SEQ_LEN   = X_train.shape[1]
    IMG_SIZE  = X_train.shape[2]
    SAVE_PATH = "best_fall_detection_model.h5"

    print(f"\n--- KERAS CNN+LSTM | Input: {X_train.shape[1:]} ---")
    model = Sequential([
        TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu'),
                        input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2,2))),
        TimeDistributed(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')),
        TimeDistributed(tf.keras.layers.MaxPooling2D((2,2))),
        TimeDistributed(tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')),
        TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
        LSTM(64), Dropout(0.5),
        Dense(32, activation='relu'), Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(SAVE_PATH, monitor='val_loss', save_best_only=True)
    ]
    history_obj = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=8, callbacks=callbacks)
    history = {"train_loss": history_obj.history['loss'],
               "val_loss":   history_obj.history['val_loss'],
               "train_acc":  history_obj.history['accuracy'],
               "val_acc":    history_obj.history['val_accuracy']}
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=========================================")
    print(f"=> DO CHINH XAC TREN TAP TEST: {acc*100:.2f}%")
    print(f"=========================================")

# ── Biểu đồ ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_acc"], label="Train Accuracy", color="blue")
axes[0].plot(history["val_acc"],   label="Val Accuracy",   color="orange")
axes[0].set_title("Accuracy"); axes[0].legend()

axes[1].plot(history["train_loss"], label="Train Loss", color="blue")
axes[1].plot(history["val_loss"],   label="Val Loss",   color="orange")
axes[1].set_title("Loss"); axes[1].legend()

plt.tight_layout()
plt.show()
