import os
import sys

# Fix encoding cho Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Token Kaggle API — set bien moi truong truoc khi chay:
#   Windows: $env:KAGGLE_API_TOKEN = "your_token_here"
#   Linux/Mac: export KAGGLE_API_TOKEN=your_token_here
# Hoac tao file ~/.kaggle/kaggle.json theo huong dan: https://www.kaggle.com/docs/api
if not os.environ.get('KAGGLE_API_TOKEN'):
    print("Loi: Chua set KAGGLE_API_TOKEN. Xem huong dan trong file nay.")
    exit(1)

import kaggle

PROJECT_DIR = "D:/Project/falldetection_model-2"
os.makedirs(PROJECT_DIR, exist_ok=True)

print("Dang ket noi Kaggle va tai dataset (khoang 100MB)...")

kaggle.api.dataset_download_files(
    'shahliza27/ur-fall-detection-dataset',
    path=PROJECT_DIR,
    unzip=True
)

print("Da tai va giai nen thanh cong!")

root = os.path.join(PROJECT_DIR, "UR_fall_detection_dataset_cam0_rgb")
if os.path.exists(root):
    print("Danh sach thu muc goc:", os.listdir(root)[:5])
else:
    print("Khong tim thay thu muc:", root)
