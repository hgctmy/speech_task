from tqdm import tqdm
import logging
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import librosa
import os
import torch
import ast
<<<<<<< HEAD
from math import floor, ceil
=======
>>>>>>> 21e48d60dad9a8388e2999e1ae46cc02571d1bc7

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_audio(audio_file, timestamps_data, labels_data, segments, output_dir, wavlm_model_name='microsoft/wavlm-base', sample_rate=16000):
    """
    音声ファイルの前処理を行い、発話ごとの単語ベクトルとラベルを保存

    Args:
        audio_file (str): 音声ファイルのパス
<<<<<<< HEAD
        timestamps_data (list): タイムスタンプデータのリスト
        labels_data (list): ラベルデータのリスト
        segments (list): 発話のタイムスタンプのリスト
=======
        timestamps_data (list): 発話ごとの単語のタイムスタンプデータのリスト
        labels_data (list): ラベルデータのリスト
        segments(list): 発話のタイムスタンプ
>>>>>>> 21e48d60dad9a8388e2999e1ae46cc02571d1bc7
        output_dir (str): 出力ディレクトリのパス
        wavlm_model_name (str): 使用するWavLMモデルの名前
        sample_rate (int): サンプリングレート
    """
    try:
<<<<<<< HEAD
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # モデル読み込み時にGPUへ転送
        wavlm_model = WavLMModel.from_pretrained(wavlm_model_name).to(device)
=======
        # WavLMモデルと特徴抽出器のロード
        wavlm_model = WavLMModel.from_pretrained(wavlm_model_name)
>>>>>>> 21e48d60dad9a8388e2999e1ae46cc02571d1bc7
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
        wavlm_model.eval()

        # 音声ファイルの読み込み
        waveform, sr = librosa.load(audio_file, sr=sample_rate)
        if len(waveform.shape) == 2:
            waveform = waveform.mean(axis=1)

<<<<<<< HEAD
        basename = timestamps_data[0]
        utterances = timestamps_data[1:]
        segments_list = segments[1:]
        labels_list = labels_data[1:]
        # データの整合性チェック
        if len(segments_list) != len(utterances) or len(utterances) != len(labels_list):
            logger.error(f"Data mismatch: segments({len(segments_list)}), utterances({len(utterances)}), labels({len(labels_list)})")
            return

        for i, (segment, utterance, labels) in enumerate(zip(segments_list, utterances, labels_list)):
            # セグメントの時間情報を取得
            start_time = segment['begin_time'] / 1000
            end_time = segment['end_time'] / 1000

            # サンプル数に変換
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            start_sample = max(0, int(start_time * sample_rate))
            end_sample = min(len(waveform), int(end_time * sample_rate))
            
            # 有効なサンプル範囲チェック
            if start_sample >= len(waveform) or end_sample > len(waveform):
                logger.warning(f"Invalid segment: {start_time}-{end_time} in {basename}, endsample:{end_sample}, waveformlength{len(waveform)}")
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

                # セグメント内での相対時間に変換しクリップ
                rel_start = max(0, word_start - start_time)
                rel_end = min(end_time - start_time, word_end - start_time)

                # 有効な時間範囲チェック（完全にセグメント外の場合のみスキップ）
                if rel_end <= 0 or rel_start >= (end_time - start_time):
                    logger.warning(f"Word completely outside segment: {word_info['Word']} ({word_start}-{word_end}) in segment {start_time}-{end_time}")
                    continue

                # フレームインデックス計算
                # wavlm_start = int(floor(rel_start / 0.02))
                # wavlm_end = int(ceil(rel_end / 0.02))
                wavlm_start = int(round(rel_start / 20))
                wavlm_end = int(round(rel_end / 20))

                # インデックスのクリッピング（最終チェック）
                wavlm_start = max(0, wavlm_start)
                wavlm_end = min(len(wavlm_output) - 1, wavlm_end)

                # インデックスが無効な場合の最終チェック
                if wavlm_start > wavlm_end:
                    logger.warning("Invalid segment")
                    continue
                # 特徴量の抽出
                elif wavlm_start == wavlm_end:
                    # 単一フレームを取得
                    word_vec = wavlm_output[wavlm_start]
                else:
                    # フレームの平均を取得
                    word_vec = wavlm_output[wavlm_start:wavlm_end+1].mean(dim=0)
                
                utterance_vectors.append(word_vec)
                utterance_labels.append(label)


            if utterance_vectors:
                utterance_data = {
                    'basename': basename,
                    'vectors': torch.stack(utterance_vectors),
                    'labels': torch.tensor(utterance_labels)
                }

                # 結果の保存
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{basename}-{i}.pt")
                torch.save([utterance_data], output_path)

    except Exception as e:
        logger.error(f"Error processing {audio_file}: {str(e)}")


# データ読み込みと処理の実行
=======
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


>>>>>>> 21e48d60dad9a8388e2999e1ae46cc02571d1bc7
audio_dir = 'audio_5700_train_dev'
segments_file = 'segments.txt'
timestamps_file = 'word_time.txt'
labels_file = 'important_word_train_clean.txt'
<<<<<<< HEAD
output_dir = "preprocessed_data_pooling_v2"

# ファイル読み込み
with open(timestamps_file, 'r', encoding='utf-8') as f:
    timestamps_data = [ast.literal_eval(line) for line in f]

with open(segments_file, 'r', encoding='utf-8') as f:
    segments_data = [ast.literal_eval(line) for line in f]

with open(labels_file, 'r', encoding='utf-8') as f:
    labels_data = [ast.literal_eval(line) for line in f]


audio_files = sorted(
    [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".wav")],
    key=lambda x: os.path.basename(x).split('.')[0]
)

segments_data.sort(key=lambda x: x[0])
labels_data.sort(key=lambda x: x[0])
timestamps_data.sort(key=lambda x: x[0])


'''
ファイルの整合性チェック
import os

def synchronize_lists(audio_files, segments_data):
    """
    音声ファイルリストとセグメントデータリストを同期化し、
    両方に存在する要素のみを残す。

    Args:
        audio_files (list): 音声ファイルパスのリスト
        segments_data (list): セグメントデータのリスト

    Returns:
        tuple: 同期化された音声ファイルリストとセグメントデータリスト
    """
    audio_basenames = set()
    for audio_file in audio_files:
        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
        audio_basenames.add(audio_basename)

    segment_basenames = set()
    for segment_item in segments_data:
        segment_basename = segment_item[0]
        segment_basenames.add(segment_basename)

    # 2. 両方のリストに共通するベース名のセット（積集合）を求める
    common_basenames = audio_basenames.intersection(segment_basenames)

    # 3. 共通するベース名を持つ要素のみを各リストから抽出して、新しいリストを作成

    synchronized_audio_files = []
    synchronized_segments_data = []

    for audio_file in audio_files:
        audio_basename = os.path.splitext(os.path.basename(audio_file))[0]
        if audio_basename in common_basenames:
            synchronized_audio_files.append(audio_file)
        else:
            print(f"audio {audio_file}")

    for segment_item in segments_data:
        segment_basename = segment_item[0]
        if segment_basename in common_basenames:
            synchronized_segments_data.append(segment_item)
        else:
            print(f"segment {segment_item[0]}")

    # 4. 同期化されたリストを返す
    return synchronized_audio_files, synchronized_segments_data


audio_files, segments_data = synchronize_lists(audio_files, segments_data)

for seg_data, audio_file in zip(segments_data, audio_files):
    parts = audio_file.split('/')
    file_name_with_extension = parts[-1].split('.')[0]
    audio_length = librosa.get_duration(filename=audio_file)
    if file_name_with_extension != seg_data[0]:
        print(f'file mismatch{file_name_with_extension},{seg_data[0]}')
    for segment in seg_data[1:]:
        end_time = segment['end_time'] / 1000.0
        if end_time > audio_length:
            print(f"Invalid segment in {seg_data[0]},filename:{audio_file}: {end_time}s > {audio_length}s")
'''

# 音声ファイル処理
for audio_file, ts_data, lb_data, seg_data in tqdm(
    zip(audio_files, timestamps_data, labels_data, segments_data),
    desc="Processing Audio Files",
    total=len(audio_files)
):
    preprocess_audio(audio_file, ts_data, lb_data, seg_data, output_dir)
=======
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
>>>>>>> 21e48d60dad9a8388e2999e1ae46cc02571d1bc7
