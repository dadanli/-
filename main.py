# main.py
"""
用法：
    python main.py image_train   -- 训练静态手势分类器
    python main.py image_infer   -- 启动实时摄像头手语字母识别
"""
import os
import sys
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#import gradio as gr

from model import Config, Vocabulary, SignDataset, SignTranslator, SignClassifier, ImageConfig

# ---------- 全局设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- MediaPipe Tasks 工具函数（供实时识别使用） ----------
def init_hand_detector():
    """初始化 HandLandmarker，返回 detector 对象"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"hand_landmarker.task 未找到！请下载后放在项目根目录。")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def extract_landmarks_live(frame, detector):
    """实时提取单帧关键点，返回 (63,) 或 None"""
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    wrist = np.array([hand[0].x, hand[0].y, hand[0].z])
    vec = []
    for lm in hand:
        p = np.array([lm.x, lm.y, lm.z]) - wrist
        vec.extend(p)
    return np.array(vec, dtype=np.float32)

# ---------- 静态手势分类训练 ----------
def image_train():
    config = ImageConfig()
    print(f"使用设备: {device}")

    # 加载数据
    train_data = np.load(config.train_data)
    X_train, y_train = train_data["X"], train_data["y"]
    val_data = np.load(config.val_data)
    X_val, y_val = val_data["X"], val_data["y"]

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    # 模型
    model = SignClassifier(input_dim=config.input_dim, num_classes=config.num_classes).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()

        acc = correct / total
        print(f"Epoch {epoch:2d}/{config.epochs} | Loss {train_loss/len(train_loader):.4f} | Val Acc {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.checkpoint_path)
            print(f"  ✓ 保存最佳模型 (acc={acc:.4f})")

    print(f"训练完成，最佳验证准确率: {best_acc:.4f}")

# ---------- 实时摄像头识别 ----------
def image_infer():
    config = ImageConfig()
    if not os.path.exists(config.checkpoint_path):
        print(f"未找到模型 {config.checkpoint_path}，请先训练。")
        sys.exit(1)

    # 加载类别映射
    with open(config.vocab_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {v: k for k, v in label2idx.items()}

    # 加载模型
    model = SignClassifier(input_dim=config.input_dim, num_classes=config.num_classes).to(device)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()
    print("模型加载成功！请在弹出的摄像头窗口中查看识别结果。")

    # 初始化手部检测器
    detector = init_hand_detector()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    print("按 Q 键退出程序。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 镜像翻转（让手势方向自然）
        frame = cv2.flip(frame, 1)

        # 提取手部关键点并推理
        landmarks = extract_landmarks_live(frame, detector)
        if landmarks is not None:
            inp = torch.FloatTensor(landmarks).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(inp)
                _, pred = torch.max(outputs, 1)
            letter = idx2label[pred.item()]
            # 在画面上显示识别结果
            cv2.putText(frame, letter, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "未检测到手部", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示画面
        cv2.imshow("Hand Sign Recognition - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- 旧版连续手语翻译函数（保留，暂时不用） ----------
def train():
    # 此处省略，如果你需要训练连续手语模型，可以恢复之前的 train() 代码
    pass

def infer():
    # 此处省略
    pass

# ---------- 主入口 ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请指定运行模式：")
        print("  image_train    训练静态手势分类器")
        print("  image_infer    启动实时摄像头识别")
    elif sys.argv[1] == "image_train":
        image_train()
    elif sys.argv[1] == "image_infer":
        image_infer()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "infer":
        infer()
    else:
        print(f"未知模式: {sys.argv[1]}")