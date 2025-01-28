import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import os
from schedulefree import RAdamScheduleFree
from tqdm import tqdm
import torch.nn.functional as F


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        seq_len = x.size(0)
        position_ids = self.position_ids[:seq_len]
        return x + self.pos_emb(position_ids).unsqueeze(1)


class ImportantWordClassifier(nn.Module):
    def __init__(self, input_dim=768, d_model=768, nhead=8, num_layers=6, max_seq_len=512):
        super().__init__()
        self.d_model = d_model

        # 入力処理
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        ) if input_dim != d_model else nn.Identity()

        # 位置埋め込み
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=max_seq_len)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            activation='gelu',
            norm_first=True,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, attention_mask=None):
        x = self.input_proj(x)

        x = x.permute(1, 0, 2)  # (seq_len, batch, dim)
        x = self.pos_encoder(x)

        src_key_padding_mask = ~attention_mask.to(torch.bool) if attention_mask is not None else None

        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = x.permute(1, 0, 2)  # (batch, seq_len, dim)
        return self.classifier(x).squeeze(-1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def calculate_metrics(preds, labels, mask, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    tp = ((preds == 1) & (labels == 1) & mask).sum()
    fp = ((preds == 1) & (labels == 0) & mask).sum()
    fn = ((preds == 0) & (labels == 1) & mask).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return precision, recall, f1


# データセットクラス
class ImportantWordDataset(Dataset):
    def __init__(self, preprocessed_dir):
        self.preprocessed_dir = preprocessed_dir
        self.data = []
        for file_name in os.listdir(preprocessed_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(preprocessed_dir, file_name)
                self.data.extend(torch.load(file_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        utterance = self.data[idx]
        return utterance['vectors'], utterance['labels']

# データ整形関数


def collate_fn(batch):
    vectors, labels = zip(*batch)
    lengths = [v.size(0) for v in vectors]
    max_len = max(lengths)

    padded_vectors = torch.zeros(len(vectors), max_len, vectors[0].size(1))
    padded_labels = torch.zeros(len(vectors), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(vectors), max_len, dtype=torch.bool)

    for i, (v, l) in enumerate(zip(vectors, labels)):
        padded_vectors[i, :lengths[i]] = v
        padded_labels[i, :lengths[i]] = l
        attention_mask[i, :lengths[i]] = 1

    return padded_vectors, padded_labels, attention_mask


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データローダー
    train_dataset = ImportantWordDataset('preprocessed_data_pooling_v3')
    # val_dataset = ImportantWordDataset('preprocessed_data/val')

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn)

    # モデル初期化
    model = ImportantWordClassifier().to(device)
    optimizer = RAdamScheduleFree(model.parameters(), lr=1e-4)
    criterion = FocalLoss(alpha=0.5, gamma=2)

    best_f1 = 0
    for epoch in range(10):
        model.train()
        optimizer.train()
        train_loss = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for vectors, labels, mask in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            vectors = vectors.to(device)
            labels = labels.to(device).float()
            mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(vectors, attention_mask=mask)

            loss = criterion(logits[mask], labels[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            # メトリクス計算（バッチごと）
            with torch.no_grad():
                batch_tp, batch_fp, batch_fn = calculate_metrics(logits, labels, mask)
                total_tp += batch_tp
                total_fp += batch_fp
                total_fn += batch_fn

        # エポック全体のメトリクス計算
        precision = total_tp / (total_tp + total_fp + 1e-12)
        recall = total_tp / (total_tp + total_fn + 1e-12)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f} | F1: {f1:.4f}')

        '''
        # 検証フェーズ
        model.eval()
        optimizer.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for vectors, labels, mask in val_loader:
                vectors = vectors.to(device)
                labels = labels.to(device).float()
                mask = mask.to(device)
                
                logits = model(vectors, attention_mask=mask)
                loss = criterion(logits[mask], labels[mask])
                
                val_loss += loss.item()
                val_preds.append(logits.cpu())
                val_labels.append(labels.cpu())
        '''

        # モデル保存
        '''if val_f1 > best_f1:
            best_f1 = val_f1'''
        torch.save(model.state_dict(), 'best_model.pth')


if __name__ == '__main__':
    train_model()
