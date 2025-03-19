import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


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


class RawDataset(Dataset):
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.data = []
        
        # データ構造を確認するためのフラグ
        self.checked_structure = False
        
        for file_name in os.listdir(raw_data_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(raw_data_dir, file_name)
                data = torch.load(file_path)
                
                # 最初のファイルのデータ構造を確認
                if not self.checked_structure and len(data) > 0:
                    print(f"Loading data from {file_path}")
                    print(f"Data structure keys: {list(data[0].keys())}")
                    self.checked_structure = True
                
                self.data.extend(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 'id'キーがなければファイル名+インデックスで代用
        item_id = item.get('id', None)
        if item_id is None:
            # 代替IDとして一意の識別子を生成
            item_id = f"item_{idx}"
        
        # データにvectorsキーがあることを確認
        if 'vectors' not in item:
            # データ構造を表示
            print(f"Item keys: {list(item.keys())}")
            raise KeyError(f"'vectors' key not found in item at index {idx}")
            
        return item['vectors'], item_id


def collate_fn(batch):
    vectors, ids = zip(*batch)
    lengths = [v.size(0) for v in vectors]
    max_len = max(lengths)

    padded_vectors = torch.zeros(len(vectors), max_len, vectors[0].size(1))
    attention_mask = torch.zeros(len(vectors), max_len, dtype=torch.bool)

    for i, v in enumerate(vectors):
        padded_vectors[i, :lengths[i]] = v
        attention_mask[i, :lengths[i]] = 1

    return padded_vectors, attention_mask, ids, lengths


def run_inference(model_path, raw_data_dir, output_dir, batch_size=8):
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 最適化されたモデルパラメータの読み込み
    checkpoint = torch.load(model_path, map_location=device)
    
    # モデルの設定 (ベストパラメータを使用)
    d_model = 1024  # 最適化されたパラメータ: d_model_base(128) * nhead(8)
    nhead = 8
    num_layers = 6
    
    model = ImportantWordClassifier(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    
    # 状態辞書のロード方法を確認
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model from checkpoint['model_state_dict']")
    else:
        # モデルの状態辞書が直接保存されている場合
        model.load_state_dict(checkpoint)
        print("Loaded model directly from checkpoint")
        
    model.eval()
    
    # データセットとデータローダーの設定
    try:
        dataset = RawDataset(raw_data_dir)
        print(f"Dataset loaded with {len(dataset)} samples")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # 推論実行
        processed_data = []
        batch_count = 0
        
        with torch.no_grad():
            for vectors, attention_mask, ids, lengths in tqdm(dataloader, desc="Inferencing"):
                vectors = vectors.to(device)
                attention_mask = attention_mask.to(device)
                
                # 予測実行（ロジットを直接取得）
                logits = model(vectors, attention_mask=attention_mask)
                
                # バッチ内の各サンプルについて処理
                for i in range(len(ids)):
                    seq_len = lengths[i]
                    
                    # 元のベクトルとIDを取得
                    original_vectors = vectors[i, :seq_len].cpu()
                    utterance_id = ids[i]
                    
                    # 重要度のロジット値を取得（連続値のまま）
                    importance_logits = logits[i, :seq_len].cpu()
                    
                    # サンプルごとの処理結果を保存
                    processed_data.append({
                        'id': utterance_id,
                        'vectors': original_vectors,
                        'importance_logits': importance_logits  # ロジット値（連続値）として保存
                    })
                
                # バッチが一定数に達したら保存
                if len(processed_data) >= 1000:
                    save_batch(processed_data, output_dir, batch_count)
                    batch_count += 1
                    processed_data = []
        
        # 残りのデータを保存
        if processed_data:
            save_batch(processed_data, output_dir, batch_count)
        
        print(f"Inference completed. Processed data saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


def save_batch(data, output_dir, batch_id):
    """データをバッチごとに保存する関数"""
    output_path = os.path.join(output_dir, f"batch_{batch_id:04d}.pt")
    torch.save(data, output_path)
    print(f"Saved {len(data)} samples to {output_path}")


def examine_data_structure(data_dir):
    """データディレクトリ内のファイルの構造を調査する関数"""
    print(f"Examining data in: {data_dir}")
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.pt'):
            file_path = os.path.join(data_dir, file_name)
            try:
                data = torch.load(file_path)
                if isinstance(data, list) and len(data) > 0:
                    print(f"\nFile: {file_name}")
                    print(f"Number of items: {len(data)}")
                    print(f"First item keys: {list(data[0].keys())}")
                    
                    # ベクトルの形状を確認
                    if 'vectors' in data[0]:
                        print(f"Vectors shape: {data[0]['vectors'].shape}")
                    
                    # 最初の数個のアイテムのみ表示
                    max_items = min(3, len(data))
                    for i in range(max_items):
                        print(f"\nItem {i} details:")
                        for k, v in data[i].items():
                            if isinstance(v, torch.Tensor):
                                print(f"  {k}: Tensor of shape {v.shape}")
                            else:
                                print(f"  {k}: {type(v)}")
                    
                    # 1ファイルの確認で十分
                    break
                else:
                    print(f"\nFile: {file_name} - Empty or not a list")
            except Exception as e:
                print(f"\nError loading {file_name}: {str(e)}")


if __name__ == "__main__":
    # パラメータ設定
    MODEL_PATH = "best_model_tuned_ones_v2.pth"  # 訓練済みモデルのパス
    RAW_DATA_DIR = "preprocessed_data_pooling"  # 生データのディレクトリ
    OUTPUT_DIR = "important_word_data"  # 処理後のデータを保存するディレクトリ
    BATCH_SIZE = 8  # バッチサイズ
    
    # データ構造の確認
    print("Examining raw data structure...")
    examine_data_structure(RAW_DATA_DIR)
    
    # 推論実行
    print("\nStarting inference...")
    run_inference(MODEL_PATH, RAW_DATA_DIR, OUTPUT_DIR, BATCH_SIZE)