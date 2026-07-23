# preprocess.py
"""
将 ASL Alphabet 图片数据集转换为手部关键点向量，并保存为 .npz 文件。
数据组织结构：
    data/raw/asl_alphabet/
        A/
            img1.jpg
            ...
        B/
        ...
        space/
        nothing/
运行后会在 data/processed 下生成 train.npz, val.npz，以及 data/label_map.json。

⚠️ 需要提前下载 hand_landmarker.task 并放在项目根目录。
"""
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------- MediaPipe Tasks API 初始化（全局） ----------
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# 模型文件路径（请确保 hand_landmarker.task 在项目根目录下）
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件 {MODEL_PATH} 未找到，请先下载！\n"
                            f"下载地址：https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def extract_landmarks(frame):
    """
    使用 MediaPipe Tasks API 从一帧图像中提取单手关键点 (63 维)
    返回：numpy array (63,) 或 None
    """
    # 转为 RGB 并构建 MediaPipe Image 对象
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # 检测手部关键点
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None

    # 只取第一只手
    hand = result.hand_landmarks[0]

    # 以手腕（第 0 个关键点）为原点，归一化坐标
    wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
    vec = []
    for lm in hand:
        p = np.array([lm.x, lm.y, lm.z]) - wrist
        vec.extend(p)
    return np.array(vec, dtype=np.float32)  # shape (63,)

def build_dataset(raw_dir, output_dir, label_map_path, test_size=0.1):
    """
    raw_dir: 包含子文件夹 A, B, C, ..., space, nothing 的根目录
    """
    # 收集所有类别文件夹
    class_names = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    print(f"发现 {len(class_names)} 个类别: {class_names}")

    # 生成类别索引映射并保存
    label2idx = {name: idx for idx, name in enumerate(class_names)}
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label2idx, f, ensure_ascii=False, indent=2)

    X, y = [], []
    skipped = 0

    for cls_name in class_names:
        cls_dir = os.path.join(raw_dir, cls_name)
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_file in tqdm(img_files, desc=f"处理 {cls_name}"):
            img_path = os.path.join(cls_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                X.append(landmarks)
                y.append(label2idx[cls_name])
            else:
                skipped += 1

    X = np.array(X, dtype=np.float32)   # (N, 63)
    y = np.array(y, dtype=np.int64)     # (N,)
    print(f"有效样本: {len(X)}, 跳过: {skipped}")

    # 分层抽样划分训练/验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                  random_state=42)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "train.npz"), X=X_train, y=y_train)
    np.savez(os.path.join(output_dir, "val.npz"), X=X_val, y=y_val)

    print(f"训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本")
    print(f"标签映射已保存至 {label_map_path}")

if __name__ == "__main__":
    # ========== 请修改这里的路径 ==========
    raw_dir = "data/raw/asl_alphabet"            # 原始图片的根目录
    output_dir = "data/processed"
    label_map_path = "data/label_map.json"
    # ====================================
    build_dataset(raw_dir, output_dir, label_map_path)