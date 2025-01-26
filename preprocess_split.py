from tqdm import tqdm
import logging
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import librosa
import os
import torch
import ast

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_audio(audio_file, timestamps_data, labels_data, segments, output_dir, wavlm_model_name='microsoft/wavlm-base', sample_rate=16000):
    """
    音声ファイルの前処理を行い、発話ごとの単語ベクトルとラベルを保存

    Args:
        audio_file (str): 音声ファイルのパス
        timestamps_data (list): タイムスタンプデータのリスト
        labels_data (list): ラベルデータのリスト
        segments (list): 発話のタイムスタンプのリスト
        output_dir (str): 出力ディレクトリのパス
        wavlm_model_name (str): 使用するWavLMモデルの名前
        sample_rate (int): サンプリングレート
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # モデル読み込み時にGPUへ転送
        wavlm_model = WavLMModel.from_pretrained(wavlm_model_name).to(device)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
        wavlm_model.eval()

        # 音声ファイルの読み込み
        waveform, sr = librosa.load(audio_file, sr=sample_rate)
        if len(waveform.shape) == 2:
            waveform = waveform.mean(axis=1)

        basename = timestamps_data[0]
        utterances = timestamps_data[1:]
        segments_list = segments[1:]
        labels_list = labels_data[1:]
        # データの整合性チェック
        if len(segments_list) != len(utterances) or len(utterances) != len(labels_list):
            logger.error(f"Data mismatch: segments({len(segments_list)}), utterances({len(utterances)}), labels({len(labels_list)})")
            return

        all_utterance_data = []
        for segment, utterance, labels in zip(segments_list, utterances, labels_list):
            # セグメントの時間情報を取得
            start_time = segment['begin_time'] / 1000  # ミリ秒から秒に変換
            end_time = segment['end_time'] / 1000

            # サンプル数に変換
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)

            # 有効なサンプル範囲チェック
            if start_sample >= len(waveform) or end_sample > len(waveform):
                logger.warning(f"Invalid segment: {start_time}-{end_time} in {basename}")
                continue

            # 発話単位で音声を切り出し
            utterance_waveform = waveform[start_sample:end_sample]
            if len(utterance_waveform) == 0:
                logger.warning(f"Empty segment: {start_time}-{end_time} in {basename}")
                continue

            # 特徴量抽出
            encoding = feature_extractor(
                utterance_waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=False
            ).to(device)

            # WavLMで特徴量取得
            with torch.no_grad():
                outputs = wavlm_model(**encoding)
                wavlm_output = outputs.last_hidden_state.squeeze(0)

            utterance_vectors = []
            utterance_labels = []
            for word_info, label in zip(utterance, labels):
                # 単語の絶対時間を取得
                word_start = word_info['BeginTime'] / 1000
                word_end = word_info['EndTime'] / 1000

                # セグメント内での相対時間に変換
                rel_start = word_start - start_time
                rel_end = word_end - start_time

                # サンプル数に変換
                start_in_utterance = int(rel_start * sample_rate)
                end_in_utterance = int(rel_end * sample_rate)

                # WavLMの出力インデックスに変換（20msごとの特徴量）
                wavlm_start = int(start_in_utterance / (0.02 * sample_rate))
                wavlm_end = int(end_in_utterance / (0.02 * sample_rate))

                # インデックスの有効性チェック
                if wavlm_start >= wavlm_end or wavlm_end > len(wavlm_output):
                    logger.warning(f"Invalid word indices: {word_info['Word']} in {basename}")
                    continue

                # 特徴量の平均を取得
                word_vec = wavlm_output[wavlm_start:wavlm_end].mean(dim=0)
                utterance_vectors.append(word_vec)
                utterance_labels.append(label)

            if utterance_vectors:
                all_utterance_data.append({
                    'basename': basename,
                    'vectors': torch.stack(utterance_vectors),
                    'labels': torch.tensor(utterance_labels)
                })

        # 結果の保存
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{basename}.pt")
        torch.save(all_utterance_data, output_path)
        logger.info(f"Saved processed data: {output_path}")

    except Exception as e:
        logger.error(f"Error processing {audio_file}: {str(e)}")


# データ読み込みと処理の実行
audio_dir = 'audio_5700_train_dev'
segments_file = 'segments.txt'
timestamps_file = 'word_time.txt'
labels_file = 'important_word_train_clean.txt'
output_dir = "preprocessed_data"

# ファイル読み込み
with open(timestamps_file, 'r', encoding='utf-8') as f:
    timestamps_data = [ast.literal_eval(line) for line in f]

with open(segments_file, 'r', encoding='utf-8') as f:
    segments_data = [ast.literal_eval(line) for line in f]

with open(labels_file, 'r', encoding='utf-8') as f:
    labels_data = [ast.literal_eval(line) for line in f]

# 音声ファイル処理
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
for audio_file, ts_data, lb_data, seg_data in tqdm(
    zip(audio_files, timestamps_data, labels_data, segments_data),
    desc="Processing Audio Files",
    total=len(audio_files)
):
    preprocess_audio(audio_file, ts_data, lb_data, seg_data, output_dir)
