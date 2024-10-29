import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
import json
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WavLMモデルとフィーチャー抽出器の初期化
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)


def get_wavlm_embeddings(audio):
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_values
    inputs = inputs.to(device)
    inputs = inputs.squeeze(1)
    inputs = inputs.squeeze(0)
    with torch.no_grad():
        outputs = wavlm_model(inputs)
    return outputs.last_hidden_state


def process_audio_file(file_path, file_name):
    # オーディオファイルをロード
    audio, sr = torchaudio.load(file_path)

    # リサンプリングが必要な場合
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
        sr = 16000

    # 各単語のタイムスタンプを取得する（ここでは簡略化のため、各フレームに対応する単語を推定）
    with open("train_output_word_time.jsonl", "r") as f:
        data = f.readlines()
        utt_list = {}
        i = 0
        for line in data:
            line = json.loads(line)
            if file_name[:7] in line["basename"]:
                word_timestamps = []
                for word in line["last_user_utt"]:
                    start, end = word["BeginTime"], word["EndTime"]
                    word_timestamps.append((start, end))

                # 開始時間と終了時間をサンプル数に変換
                begin_sample = int(word_timestamps[0][0] / 1000 * sr)
                end_sample = int(word_timestamps[-1][1] / 1000 * sr)

                # 音声ファイルの範囲を抽出
                segment = audio[:, begin_sample:end_sample]

                # オーディオデータの次元を調整
                segment = segment.unsqueeze(0)  # バッチ次元を追加

                # 埋め込みを取得
                embeddings = get_wavlm_embeddings(segment)

                frame_duration = (word_timestamps[-1][1] / 1000 - word_timestamps[0][0] / 1000) / embeddings.size(1)

                # 各単語ごとの埋め込みを取得
                word_embeddings = []
                for start, end in word_timestamps:
                    start_frame = int(((start / 1000) - (word_timestamps[0][0] / 1000)) / frame_duration)
                    end_frame = int(((end / 1000) - (word_timestamps[0][0] / 1000)) / frame_duration)
                    word_embedding = embeddings[0, start_frame:end_frame].mean(dim=0)
                    word_embeddings.append(word_embedding.tolist())
                utt_list[file_name + str(i * 2)] = word_embeddings
                i += 1
    return utt_list


# フォルダ内のすべてのWAVファイルを処理
folder_path = "audio_5700_train_dev"
output_embeddings = {}

for file_name in tqdm(os.listdir(folder_path)):
    if file_name.endswith(".wav"):
        file_path = os.path.join(folder_path, file_name)
        embeddings = process_audio_file(file_path, file_name)
        output_embeddings[file_name] = embeddings

# 結果をJSONファイルに保存
with open("embeddings_train.json", "w") as f:
    json.dump(output_embeddings, f, indent=4)

print("Embeddings have been saved to embeddings.json")
