# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

# ---------- 全局配置 ----------
class Config:
    # 数据
    input_dim = 63              # MediaPipe 单手 21 个关键点 * 3 坐标
    max_seq_len = 200           # 最大帧数
    vocab_path = "data/vocab.json"

    # 模型结构
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.1

    # 训练超参
    batch_size = 8             # 根据显存调整（12GB → 16; 8GB → 8; 6GB → 4）
    epochs = 50
    lr = 1e-3
    weight_decay = 1e-4
    grad_clip = 1.0

    # 路径
    train_data = "data/processed/train.npz"
    val_data = "data/processed/val.npz"
    checkpoint_dir = "checkpoints"

# ---------- 词汇表 ----------
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}

    def add_sentence(self, sentence):
        # sentence 是字符列表（中文逐字）
        for ch in sentence:
            if ch not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[ch] = idx
                self.idx2word[idx] = ch

    def encode(self, sentence):
        if isinstance(sentence, str):
            sentence = list(sentence)
        return [self.word2idx["<sos>"]] + \
               [self.word2idx.get(ch, self.word2idx["<unk>"]) for ch in sentence] + \
               [self.word2idx["<eos>"]]

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx in [self.word2idx["<pad>"], self.word2idx["<sos>"]]:
                continue
            if idx == self.word2idx["<eos>"]:
                break
            words.append(self.idx2word.get(idx, "<unk>"))
        return "".join(words)

    def __len__(self):
        return len(self.word2idx)

# ---------- 数据集 ----------
class SignDataset(Dataset):
    def __init__(self, npz_path, max_src_len=200, augment=False):
        data = np.load(npz_path, allow_pickle=True)
        self.src = list(data["src"])
        self.tgt = list(data["tgt"])
        self.max_src_len = max_src_len
        self.augment = augment

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = torch.FloatTensor(self.src[idx])
        tgt = torch.LongTensor(self.tgt[idx])
        if src.size(0) > self.max_src_len:
            if self.augment:
                start = random.randint(0, src.size(0) - self.max_src_len)
                src = src[start:start + self.max_src_len]
            else:
                src = src[:self.max_src_len]
        if self.augment:
            if random.random() > 0.5:
                src = src + torch.randn_like(src) * 0.02
            if random.random() > 0.5:
                scale = random.uniform(0.8, 1.2)
                new_len = max(2, int(src.size(0) * scale))
                idxs = torch.linspace(0, src.size(0)-1, new_len).long()
                src = src[idxs]
        return {"src": src, "tgt": tgt}

    @staticmethod
    def collate_fn(batch):
        B = len(batch)
        src_list = [item["src"] for item in batch]
        tgt_list = [item["tgt"] for item in batch]
        max_src = max(s.size(0) for s in src_list)
        max_tgt = max(t.size(0) for t in tgt_list)
        src_padded = torch.zeros(B, max_src, 63)
        tgt_padded = torch.zeros(B, max_tgt, dtype=torch.long)
        for i in range(B):
            src_padded[i, :src_list[i].size(0)] = src_list[i]
            tgt_padded[i, :tgt_list[i].size(0)] = tgt_list[i]
        return {"src": src_padded, "tgt": tgt_padded}

# ---------- 模型组件（连续手语翻译用，保留） ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SignTranslator(nn.Module):
    # 连续手语翻译模型，此处保留完整代码
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, src, tgt=None):
        src = src.permute(1, 0, 2)
        src = self.input_proj(src)
        src = src.permute(1, 2, 0)
        src = F.relu(self.conv1(src))
        src = F.relu(self.conv2(src))
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        if tgt is not None:
            tgt = tgt.permute(1, 0)
            tgt_emb = self.text_embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(src.device)
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            return self.fc_out(output).permute(1, 0, 2)
        else:
            return self._greedy_decode(memory, src.device)

    def _greedy_decode(self, memory, device, max_len=50):
        batch_size = memory.size(1)
        sos_idx = 1
        eos_idx = 2
        generated = torch.full((1, batch_size), sos_idx, dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_emb = self.text_embedding(generated) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(device)
            out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            prob = self.fc_out(out[-1, :, :])
            next_token = prob.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=0)
            if (next_token == eos_idx).all():
                break
        return generated.permute(1, 0)

# ---------- 静态手势分类器 ----------
class SignClassifier(nn.Module):
    def __init__(self, input_dim=63, num_classes=29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------- 图片分类配置 ----------
class ImageConfig:
    input_dim = 63
    num_classes = 29            # ASL Alphabet 共 29 个类别
    batch_size = 64
    epochs = 15
    lr = 1e-3
    train_data = "data/processed/train.npz"
    val_data   = "data/processed/val.npz"
    vocab_path = "data/label_map.json"      # 类别映射
    checkpoint_path = "checkpoints/best_classifier.pth"