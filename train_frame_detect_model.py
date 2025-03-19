import torch
from torch.utils.data import Dataset
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
from sklearn.metrics import f1_score
from transformers import Trainer, TrainingArguments
import logging
from joblib import Parallel, delayed
import librosa
import ast
from tqdm import tqdm
import torch.nn.functional as F

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# WavLMモデルを一度ロード
wavlm_model = WavLMModel.from_pretrained('microsoft/wavlm-base')


def process_segment(segment_line, label_line, audio_dir, feature_extractor, sample_rate, preprocessed_dir):
    try:
        # タイムスタンプとラベルのパース
        segment_data = ast.literal_eval(segment_line)
        audio_filename = segment_data[0]
        segments = segment_data[1:]

        label_data = ast.literal_eval(label_line)
        label_filename = label_data[0]
        labels = label_data[1:]

        assert audio_filename == label_filename, f"ファイル名が一致しません: {audio_filename} != {label_filename}"
        assert len(segments) == len(labels), f"セグメント数とラベル数が一致しません: {len(segments)} != {len(labels)}"

        # 音声ファイルの読み込み
        audio_file_path = os.path.join(audio_dir, f"{audio_filename}.wav")
        try:
            waveform, sr = librosa.load(audio_file_path, sr=sample_rate)
            logger.info(f"Loaded audio with sample rate: {sr} using librosa")
        except FileNotFoundError:
            logger.error(f"Audio file not found: {audio_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error reading audio file {audio_file_path} with librosa: {e}")
            return None

        # チャネルの処理（ステレオの場合はモノラルに変換）
        if len(waveform.shape) == 2:
            waveform = waveform.mean(axis=1)

        processed_segments = []
        for i, (segment, label) in enumerate(zip(segments, labels)):
            begin_time = segment['begin_time']  # ミリ秒(ms)
            end_time = segment['end_time']      # ミリ秒(ms)
            # セグメントの抽出
            begin_sample = int(begin_time * sr / 1000)
            end_sample = int(end_time * sr / 1000)
            segment_waveform = waveform[begin_sample:end_sample]
            # 短すぎるセグメントはスキップ
            if len(segment_waveform) < 160:
                logger.warning(f"Skipping short segment in {audio_filename} at {begin_time}-{end_time}ms")
                continue

            # 前処理を適用
            encoding = feature_extractor(
                segment_waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=False
            )
            segment_waveform_processed = encoding['input_values'].squeeze(0)

            with torch.no_grad():
                wavlm_output_length = len(wavlm_model.forward(encoding['input_values']).last_hidden_state.squeeze(0))

            # ラベルをテンソルに変換（WavLMの出力長に対応するように調整）
            if isinstance(label, list):
                if len(label) != wavlm_output_length:
                    logger.error(f"Label length does not match WavLM output length: {len(label)} vs {wavlm_output_length}")
                    continue
                label_tensor = torch.tensor(label, dtype=torch.float32)
            else:
                label_tensor = torch.full((wavlm_output_length,), float(label), dtype=torch.float32)

            processed_segments.append({
                'input_values': segment_waveform_processed,
                'labels': label_tensor,
                'filename': f"{audio_filename}_{i}.pt"
            })
        return processed_segments
    except AssertionError as e:
        logger.error(f"Assertion error: {e}")
        return None


def preprocess_and_save_data_parallel(segment_list_file, label_list_file, audio_dir, feature_extractor, sample_rate, preprocessed_dir, n_jobs=-1):
    """
    データを並列処理で前処理し、各サンプルを個別のファイルとして保存

    Args:
        segment_list_file (str): タイムスタンプのリストが含まれるファイルのパス。
        label_list_file (str): 対応するラベルのリストが含まれるファイルのパス。
        audio_dir (str): 音声ファイルが保存されているディレクトリのパス。
        feature_extractor (Wav2Vec2FeatureExtractor): Wav2Vec2の特徴抽出器。
        sample_rate (int): サンプリングレート。
        preprocessed_dir (str): 前処理済みデータを保存するディレクトリのパス。
        n_jobs (int): 並列処理に使うコア数 (-1 で全てのコアを使用)。
    """
    if not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir)

    # タイムスタンプとラベルの読み込み
    with open(segment_list_file, 'r') as f:
        segment_lines = f.readlines()
    with open(label_list_file, 'r') as f:
        label_lines = f.readlines()

    assert len(segment_lines) == len(label_lines), "タイムスタンプファイルとラベルファイルの行数が一致しません。"

    # 各セグメントを並列処理で処理
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_segment)(segment_line, label_line, audio_dir, feature_extractor, sample_rate, preprocessed_dir)
        for segment_line, label_line in zip(segment_lines, label_lines)
    )

    # 結果を保存
    for result in tqdm(results, desc="Saving preprocessed data"):
        if result is not None:
            for item in result:
                sample_path = os.path.join(preprocessed_dir, item['filename'])
                torch.save({
                    'input_values': item['input_values'],
                    'labels': item['labels']
                }, sample_path)


class SpeechSegmentDataset(Dataset):
    def __init__(self, preprocessed_dir):
        """
        Args:
            preprocessed_dir (str): 前処理済みデータファイルが保存されているディレクトリのパス。
        """
        self.preprocessed_dir = preprocessed_dir
        self.sample_files = sorted(os.listdir(preprocessed_dir))

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample_file = os.path.join(self.preprocessed_dir, self.sample_files[idx])
        sample = torch.load(sample_file)
        return sample


def data_collator(batch):
    """
    バッチ内の可変長テンソルをパディングし、アテンションマスクとラベルもパディングするコレート関数。
    ラベルのパディング方法を修正。

    Args:
        batch (list): データセットからのバッチデータのリスト。

    Returns:
        dict: パディングされた入力値、アテンションマスク、パディングされたラベル。
    """
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 入力のパディング
    input_values_padded = pad_sequence(input_values, batch_first=True)

    # アテンションマスクの作成
    attention_mask = torch.zeros(input_values_padded.shape, dtype=torch.long)
    for i, input in enumerate(input_values):
        attention_mask[i, :input.size(0)] = 1

    # ラベルのパディング（WavLMの出力長に合わせる）
    labels_padded = []
    max_label_length = 0  # バッチ内の最大ラベル長を追跡
    with torch.no_grad():
        for i, input in enumerate(input_values):
            input_length = len(wavlm_model.forward(pad_sequence([input], batch_first=True)).last_hidden_state.squeeze(0))
            max_label_length = max(max_label_length, input_length)  # 最大長を更新
    for i, label in enumerate(labels):
        label_length = len(label)
        pad_length = max_label_length - label_length
        padded_label = torch.cat([label, torch.full((pad_length,), -100.0)], dim=-1)  # dim=-1 を明示的に指定
        labels_padded.append(padded_label)

    labels_padded = torch.stack(labels_padded)

    return {
        'input_values': input_values_padded,
        'attention_mask': attention_mask,
        'labels': labels_padded
    }


def custom_bce_with_logits_loss(logits, target, weight=None, ignore_index=-100):
    mask = target != ignore_index

    mask_1d = mask.view(-1)

    masked_logits = logits.view(-1)[mask_1d]
    masked_target = target.view(-1)[mask_1d]

    masked_target = masked_target.to(logits.device)

    if weight is not None:
        weight = weight.to(logits.device)
        masked_weight = weight[masked_target.long()]
        loss = F.binary_cross_entropy_with_logits(masked_logits, masked_target, weight=masked_weight)
    else:
        loss = F.binary_cross_entropy_with_logits(masked_logits, masked_target)
    return loss


class SpeechImportanceModel(nn.Module):
    def __init__(self, wavlm_model_name='microsoft/wavlm-base-plus', dropout=0.1):
        super(SpeechImportanceModel, self).__init__()

        # 事前学習済み WavLM モデルのロード
        self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)

        # 最終的な分類層
        self.classifier = nn.Linear(self.wavlm.config.hidden_size, 1)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wavlm(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        logits = self.classifier(hidden_states).squeeze(-1)

        if labels is not None:
            # ここでweightを適用
            if hasattr(self, 'loss_weight'):
                loss = custom_bce_with_logits_loss(logits, labels, weight=self.loss_weight)
            else:
                loss = custom_bce_with_logits_loss(logits, labels)
            return {'loss': loss, 'probs': torch.sigmoid(logits)}
        else:
            return torch.sigmoid(logits)


def compute_metrics(pred):
    preds = torch.sigmoid(torch.tensor(pred.predictions))
    labels = pred.label_ids

    # パディング部分をマスク
    active = labels != -100
    preds = preds[active]
    labels = labels[active]

    f1 = f1_score(labels, (preds > 0.5).int(), average='binary')
    return {'f1': f1}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


def main():
    # パラメータの設定
    segment_list_file = 'segments.txt'  # タイムスタンプのリストが含まれるファイル
    label_list_file = 'important_frame_train.txt'      # 対応するラベルのリストが含まれるファイル
    audio_dir = 'audio_5700_train_dev'  # 音声ファイルが保存されているディレクトリ
    output_dir = './model_output'       # モデルの保存先ディレクトリ
    preprocessed_dir = './preprocessed_data'  # 前処理済みデータを保存するディレクトリ
    batch_size = 1
    sample_rate = 16000  # WavLMモデルのデフォルトサンプリングレート
    num_epochs = 5
    learning_rate = 1e-4

    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 特徴抽出器のロード
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base')

    # 前処理済みデータの存在を確認
    if not os.path.exists(preprocessed_dir) or len(os.listdir(preprocessed_dir)) == 0:
        logger.info("データを前処理しています...")
        preprocess_and_save_data_parallel(
            segment_list_file=segment_list_file,
            label_list_file=label_list_file,
            audio_dir=audio_dir,
            feature_extractor=feature_extractor,
            sample_rate=sample_rate,
            preprocessed_dir=preprocessed_dir,
            n_jobs=-1  # 並列処理に利用するコア数
        )
    else:
        logger.info("既に前処理済みのデータが見つかりました。前処理をスキップします。")

    # データセットの作成
    dataset = SpeechSegmentDataset(
        preprocessed_dir=preprocessed_dir
    )
    # クラスの重みを計算（main関数内に移動）
    labels = []
    for data in dataset:
        labels.extend(data['labels'].tolist())
    labels_tensor = torch.tensor(labels)
    positive_count = torch.sum(labels_tensor == 1).item()
    negative_count = torch.sum(labels_tensor == 0).item()
    total_count = len(labels)

    weight_positive = total_count / (2 * positive_count) if positive_count > 0 else 0.5
    weight_negative = total_count / (2 * negative_count) if negative_count > 0 else 0.5

    weight_positive = max(weight_positive, 1e-6)
    weight_negative = max(weight_negative, 1e-6)

    weights = torch.tensor([weight_negative, weight_positive]).to(device)
    # モデルの初期化
    model = SpeechImportanceModel()
    model.to(device)
    # モデルに重みを設定
    model.loss_weight = weights
    # データセットの分割（例: 80% トレーニング、20% 評価）
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    checkpoint_path = None
    if os.path.exists(output_dir):  # output_dirが存在するか確認
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # 最新のチェックポイントを取得
            checkpoint_nums = [int(cp.split("-")[1]) for cp in checkpoints]
            latest_checkpoint = f"checkpoint-{max(checkpoint_nums)}"
            checkpoint_path = os.path.join(output_dir, latest_checkpoint)
            logger.info(f"チェックポイント {checkpoint_path} からロードします。")
            # モデルとTrainerの状態をロード
            trainer = MyTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=output_dir,
                    optim="adamw_torch",
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    logging_steps=10,
                    save_steps=100,
                    evaluation_strategy="steps",
                    eval_steps=100,
                    save_total_limit=2,
                    remove_unused_columns=False,
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    fp16=torch.cuda.is_available(),
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=None,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            logger.info("チェックポイントが見つかりませんでした。新規にトレーニングを開始します。")
            trainer = MyTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=output_dir,
                    optim="adamw_torch",
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    logging_steps=10,
                    save_steps=100,
                    evaluation_strategy="steps",
                    eval_steps=100,
                    save_total_limit=2,
                    remove_unused_columns=False,
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    fp16=torch.cuda.is_available()
                ),
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=None,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
            trainer.train()
    else:  # output_dirが存在しない場合
        logger.info("出力ディレクトリが見つかりませんでした。新規にトレーニングを開始します。")
        os.makedirs(output_dir)  # output_dirを作成
        trainer = MyTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                optim="adamw_torch",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,
                save_steps=100,
                evaluation_strategy="steps",
                eval_steps=100,
                save_total_limit=2,
                remove_unused_columns=False,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                fp16=torch.cuda.is_available()
            ),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=None,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
        trainer.train()

    # モデルの保存
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
