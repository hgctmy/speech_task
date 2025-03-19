import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import os
from schedulefree import RAdamScheduleFree
from tqdm import tqdm
import torch.nn.functional as F
import optuna


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
            nn.LayerNorm(d_model),
            nn.Dropout(0.1)
        ) if input_dim != d_model else nn.Identity()

        # 位置埋め込み
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len=max_seq_len)

        # Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
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
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(d_model//2),
            nn.Linear(d_model//2, 1)
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
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

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
    padded_labels = torch.zeros(len(vectors), max_len, dtype=torch.float)
    attention_mask = torch.zeros(len(vectors), max_len, dtype=torch.bool)

    for i, (v, l) in enumerate(zip(vectors, labels)):
        padded_vectors[i, :lengths[i]] = v
        padded_labels[i, :lengths[i]] = l
        attention_mask[i, :lengths[i]] = 1

    return padded_vectors, padded_labels, attention_mask


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    optimizer.train()
    train_loss = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for vectors, labels, mask in tqdm(loader, desc=f'Training'):
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
            precision, recall, f1 = calculate_metrics(logits, labels, mask)
            total_tp += precision
            total_fp += recall
            total_fn += f1

    train_precision = total_tp / len(loader)
    train_recall = total_fp / len(loader)
    train_f1 = total_fn / len(loader)
    train_loss = train_loss / len(loader)

    return train_loss, train_f1


def validate_epoch(model, loader, optimizer, criterion, device):
    model.eval()
    optimizer.eval() # optimizer is needed for loading checkpoint, but not used in validation.
    val_loss = 0
    val_tp_total = 0
    val_fp_total = 0
    val_fn_total = 0

    with torch.no_grad():
        for vectors, labels, mask in tqdm(loader, desc=f'Validation'):
            vectors = vectors.to(device)
            labels = labels.to(device).float()
            mask = mask.to(device)

            logits = model(vectors, attention_mask=mask)
            loss = criterion(logits[mask], labels[mask])
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            val_tp_batch = ((preds == 1) & (labels == 1) & mask).sum()
            val_fp_batch = ((preds == 1) & (labels == 0) & mask).sum()
            val_fn_batch = ((preds == 0) & (labels == 1) & mask).sum()
            val_tp_total += val_tp_batch
            val_fp_total += val_fp_batch
            val_fn_total += val_fn_batch

    val_precision = val_tp_total / (val_tp_total + val_fp_total + 1e-12)
    val_recall = val_tp_total / (val_tp_total + val_fn_total + 1e-12)
    val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-12)
    val_loss = val_loss / len(loader)

    return val_loss, val_f1




def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ハイパーパラメータの提案 (範囲を狭める)
    nhead = trial.suggest_int('nhead', 2, 8) # nhead の範囲を小さく
    d_model_base = trial.suggest_int('d_model_base', 32, 128, step=32) # d_model_base の範囲を小さく
    d_model = d_model_base * nhead
    num_layers = trial.suggest_int('num_layers', 2, 4) # num_layers の範囲を小さく
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float('focal_alpha', 0.1, 0.7)
    gamma = trial.suggest_float('focal_gamma', 1.5, 3.0)
    batch_size = trial.suggest_categorical('batch_size', [16, 32]) # batch_size の候補を小さく


    # データセットの準備
    full_dataset = ImportantWordDataset('preprocessed_data_pooling')
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # モデル、オプティマイザ、損失関数の初期化
    model = ImportantWordClassifier(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    optimizer = RAdamScheduleFree(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)

    best_f1 = 0
    patience = 3
    patience_counter = 0

    for epoch in range(20): # Reduced epochs for tuning to save time. Original was 120.
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, optimizer, criterion, device)

        trial.report(val_f1, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')


        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break

    return best_f1


def train_model_with_best_params(best_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データセットの準備 (tuning と同じ設定を使用)
    full_dataset = ImportantWordDataset('preprocessed_data_pooling')
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=best_params['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_params['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # ベストパラメータでモデル、オプティマイザ、損失関数を初期化
    model = ImportantWordClassifier(
        d_model=best_params['d_model'],
        nhead=best_params['nhead'],
        num_layers=best_params['num_layers']
    ).to(device)
    optimizer = RAdamScheduleFree(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = FocalLoss(alpha=best_params['focal_alpha'], gamma=best_params['focal_gamma'])

    best_f1 = 0
    patience = 5 # Increased patience for final training
    patience_counter = 0
    start_epoch = 0
    checkpoint_path = 'best_model_tuned.pth'

    # チェックポイントの読み込み
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['best_f1']
        print(f"Checkpoint loaded. Start epoch: {start_epoch}, Best Val F1: {best_f1:.4f}")
    else:
        print("No checkpoint found. Starting training from scratch with best params.")


    for epoch in range(start_epoch, 120): # Full training epochs. Original was 120.
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = validate_epoch(model, val_loader, optimizer, criterion, device)


        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')


        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_f1': train_f1,
                'val_f1': val_f1,
                'best_f1': best_f1
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200) # Reduced trials for example. Original could be higher.

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = study.best_params
    print("\nTraining final model with best parameters...")
    
    best_params = {'nhead': 7, 'd_model': 128*7, 'num_layers': 4, 'lr': 8.528669877127414e-05, 'weight_decay': 0.0008742002971101652, 'focal_alpha': 0.6556026346175093,'focal_gamma': 1.5319969653017722, 'batch_size': 32}
    train_model_with_best_params(best_params)
    print("Final model training complete.")
