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
        timestamps_data (list): 発話ごとの単語のタイムスタンプデータのリスト
        labels_data (list): ラベルデータのリスト
        segments(list): 発話のタイムスタンプ
        output_dir (str): 出力ディレクトリのパス
        wavlm_model_name (str): 使用するWavLMモデルの名前
        sample_rate (int): サンプリングレート
    """
    try:
        # WavLMモデルと特徴抽出器のロード
        wavlm_model = WavLMModel.from_pretrained(wavlm_model_name)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
        wavlm_model.eval()

        # 音声ファイルの読み込み
        waveform, sr = librosa.load(audio_file, sr=sample_rate)
        if len(waveform.shape) == 2:
            waveform = waveform.mean(axis=1)

        # 前処理
        encoding = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=False)

        basename = timestamps_data[0]
        utterances = timestamps_data[1:]
        # 対応するラベルを取得
        if basename == labels_data[0]:
            labelslist = labels_data[1:]
        else:
            logger.warning(f"No labels found for basename: {basename}. Skipping.")
        if len(utterances) != len(labelslist):
            logger.warning(f"Number of words and labels mismatch for basename: {basename} ({len(utterances)} vs {len(labelslist)}). Skipping.")

        all_utterance_data = []
        for utterance, labels in zip(utterances, labelslist):
            utterance_vectors = []
            utterance_labels = []
            with torch.no_grad():
                wavlm_output = wavlm_model(**encoding).last_hidden_state.squeeze(0)
            for i, word_info in enumerate(utterance):
                start_time = word_info['BeginTime'] / 1000
                end_time = word_info['EndTime'] / 1000
                word = word_info['Word']
                label = labels[i]  # 対応するラベルを取得

                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                wavlm_start_index = int(start_sample / (sample_rate * 0.02))
                wavlm_end_index = int(end_sample / (sample_rate * 0.02))

                if wavlm_end_index > len(wavlm_output):
                    wavlm_end_index = len(wavlm_output)

                if wavlm_start_index >= wavlm_end_index:
                    logger.warning(f"Skipping word '{word}' in {basename} due to invalid time range: {start_time} - {end_time} (indices: {wavlm_start_index} - {wavlm_end_index}, wavlm_output length: {len(wavlm_output)}).")
                    continue

                word_vectors = wavlm_output[wavlm_start_index:wavlm_end_index]
                if len(word_vectors) == 0:
                    logger.warning(f"Skipping word '{word}' in {basename} due to no vectors.")
                    continue

                pooled_vector = word_vectors.mean(dim=0)
                utterance_vectors.append(pooled_vector)
                utterance_labels.append(label)  # ラベルを追加

            if utterance_vectors:
                utterance_vectors_tensor = torch.stack(utterance_vectors)
                utterance_labels_tensor = torch.tensor(utterance_labels)

                all_utterance_data.append({
                    'basename': basename,
                    'vectors': utterance_vectors_tensor,
                    'labels': utterance_labels_tensor  # ラベルを追加
                })

        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(output_dir, exist_ok=True)

        # データを保存
        output_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".pt"
        output_path = os.path.join(output_dir, output_filename)
        torch.save(all_utterance_data, output_path)
        logger.info(f"Preprocessed data saved to {output_path}")

    except FileNotFoundError:
        logger.error(f"File not found: {audio_file}")
    except Exception as e:
        logger.error(f"An error occurred while processing {audio_file}: {e}")


audio_dir = 'audio_5700_train_dev'
segments_file = 'segments.txt'
timestamps_file = 'word_time.txt'
labels_file = 'important_word_train_clean.txt'
output_dir = "preprocessed_data"  # 出力ディレクトリ

# タイムスタンプファイルの読み込み
with open(timestamps_file, 'r', encoding='utf-8') as f:
    timestamps_data = f.readlines()

# セグメントファイルの読み込み
with open(segments_file, 'r', encoding='utf-8') as f:
    segments_data = f.readlines()

# ラベルファイルの読み込み
with open(labels_file, 'r', encoding='utf-8') as f:
    labels_data = f.readlines()


audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")]
for audio_file_path, timestamps, labels, segments in tqdm(zip(audio_files, timestamps_data, labels_data, segments_data), desc="Processing audio files"):
    basename = os.path.splitext(os.path.basename(audio_file_path))[0]
    labels = ast.literal_eval(labels)
    timestamps = ast.literal_eval(timestamps)
    segments = ast.literal_eval(segments)
    preprocess_audio(audio_file_path, timestamps, labels, segments, output_dir)
