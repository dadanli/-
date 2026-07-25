# -```markdown
# 手语字母实时识别系统 (Sign Language Alphabet Recognition)

基于 **MediaPipe Tasks API** 与 **轻量级 MLP** 的实时手语字母识别系统。
只需一个普通摄像头，就能在本地实时识别 29 类美国手语 (ASL) 字母手势，并将结果直接显示在屏幕上。

**模型大小仅约 2 万参数，完全本地运行，无需联网，隐私友好。**

---

## ✨ 主要功能

- 📷 **实时摄像头识别**：打开摄像头即可实时识别手势，结果叠加显示在视频画面上。
- ✋ **高精度手部关键点提取**：使用 MediaPipe Hand Landmarker (Tasks API) 提取 21 个手部三维关键点，并以手腕为原点归一化。
- 🧠 **超轻量分类器**：三层全连接网络 (MLP)，输入 63 维关键点，输出 29 类字母。
- ⚡ **低延迟**：单次推理耗时 < 1ms，整体流水线延迟 < 60ms，帧率约 18 FPS（纯 CPU 环境）。
- 🔒 **隐私安全**：所有处理均在本地完成，不上传任何图像或特征。
- 🗂️ **模块化设计**：代码结构清晰，分为数据预处理、模型训练、实时推理三个独立模块。

---

## 📊 项目效果

验证集准确率 **96.2%**（训练 15 个 epoch）。

常见字母识别成功率（人工测试 50 次）：

| 字母 | A | B | C | D | E |
|------|---|---|---|---|---|
| 成功率 | 98% | 96% | 94% | 96% | 92% |

> ⚠️ 动态手势（如 J、Z）在当前静态单帧分类器下无法识别，属于设计限制。

---

## 🛠️ 环境要求

- **操作系统**：Windows / macOS / Linux
- **Python**：3.10 推荐
- **深度学习框架**：PyTorch (CPU 版即可推理，GPU 可加速训练)
- **关键点模型**：[hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) (需手动下载)
- **摄像头**：普通 USB 摄像头或笔记本内置摄像头

### 依赖库

主要依赖如下（完整列表见 `requirements.txt`）：

- `torch` (PyTorch)
- `mediapipe` (0.10.35 及以上)
- `opencv-python` (4.10.0 推荐)
- `numpy`
- `tqdm`
- `scikit-learn`

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/dadanli/-.git
cd -
```

### 2. 创建虚拟环境 (推荐)

```bash
conda create -n signlang python=3.10 -y
conda activate signlang
```

### 3. 安装依赖

```bash
# 安装 PyTorch (CPU 版)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 或者 安装 GPU 版 PyTorch (请根据 CUDA 版本选择命令)
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install mediapipe opencv-python numpy tqdm scikit-learn
```

### 4. 下载手部关键点模型

将 [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) 下载到项目**根目录**。

### 5. 准备数据集

- 从 [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) 下载数据集（`archive.zip`）。
- 解压到 `data/raw/asl_alphabet/`，确保目录结构如下：

```
data/raw/asl_alphabet/
├── A/
├── B/
├── ...
├── space/
├── nothing/
```

### 6. 数据预处理

```bash
python preprocess.py
```

该脚本会：
- 读取每张图片，提取手部关键点并归一化。
- 过滤无法检测到手的图片。
- 生成 `data/processed/train.npz`、`val.npz` 和 `data/label_map.json`。

### 7. 训练模型 (可选，如果你不想使用预训练权重)

```bash
python main.py image_train
```

默认训练 15 个 epoch，模型将保存到 `checkpoints/best_classifier.pth`。
在现代 GPU（如 RTX 5070）上训练仅需几分钟，CPU 也完全可行。

### 8. 启动实时识别

```bash
python main.py image_infer
```

- 程序会打开一个名为 `Hand Sign Recognition` 的窗口。
- 在摄像头前做出手语手势，画面会实时显示识别出的字母。
- 按键盘 **`q` 键**退出程序。

---

## 🗂️ 项目结构

```
.
├── data/
│   ├── raw/                  # 原始数据集 (需自行下载)
│   │   └── asl_alphabet/     # 按类别存放图片的文件夹
│   └── processed/            # 预处理后的关键点数据 (preprocess.py 生成)
│       ├── train.npz
│       └── val.npz
├── checkpoints/              # 训练好的模型权重 (main.py 生成)
│   └── best_classifier.pth
├── model.py                  # 模型定义、数据集类、配置
├── preprocess.py             # 数据预处理脚本
├── main.py                   # 训练和实时推理的入口
├── hand_landmarker.task      # MediaPipe 手部关键点模型 (需手动下载)
├── label_map.json            # 类别标签映射 (preprocess.py 生成)
├── requirements.txt          # 项目依赖
└── README.md                 # 本文件
```

---

## 🧠 模型说明

采用 **“关键点提取 + MLP 分类”** 的两阶段方案：

1. **关键点提取**
   MediaPipe Hand Landmarker 检测手部 21 个关键点 (x, y, z)，以手腕为原点进行归一化，得到 63 维特征向量。

2. **MLP 分类器**
   - 输入层：63 维
   - 隐藏层 1：128 维（ReLU + Dropout 0.3）
   - 隐藏层 2：64 维（ReLU + Dropout 0.3）
   - 输出层：29 维（对应 29 个类别）

模型总参数量约 **2.1 万**，体积小于 100 KB，推理速度极快，适合边缘设备部署。

---

## 🔧 命令行接口

| 命令 | 功能 |
|------|------|
| `python preprocess.py` | 从原始图片生成训练数据 |
| `python main.py image_train` | 训练 MLP 分类器 |
| `python main.py image_infer` | 启动摄像头实时识别 |

---

## 📈 测试结果

在验证集（约 8,000 样本）上的表现：

| Epoch | Train Loss | Val Accuracy |
|-------|------------|--------------|
| 1     | 2.35       | 82.1%        |
| 5     | 0.78       | 93.5%        |
| 10    | 0.42       | 95.8%        |
| 15    | 0.28       | **96.2%**    |

---

## 👥 团队成员

| 成员 | 主要贡献 | 联系方式 |
|------|----------|----------|
| 李沅轩 | 模型设计、训练代码、GPU 优化 | - |
| 王纪洲 | 数据预处理、MediaPipe 集成、部署 | - |
| 李昱全 | 测试评估、文档撰写、汇报 | - |

---

## 📝 许可证

本项目仅用于学术研究与教学目的，数据集版权归原作者所有。
代码部分采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- **数据集提供**：[Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **关键点模型**：[Google MediaPipe](https://developers.google.com/mediapipe)
- **深度学习框架**：[PyTorch](https://pytorch.org/)

---

*如果本项目对你有帮助，请给一颗 ⭐ Star，谢谢！*
```
