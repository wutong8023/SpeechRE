import pandas as pd
import soundfile as sf

audio = input("Please input audio:")
file_name = "/data/wangguitao/IWSLT/Datasets/test/" + audio
audio_input, sample_rate = sf.read(file_name)
# print("Your audio path is", file_name)

df = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
df.loc[id] = ["test-0",
              file_name + ":0:" + str(len(audio_input)),
              int(len(audio_input) / 16),
              len(audio_input),
              "<triplet>",
              "0",
              "en"
              ]

with open("/data/wangguitao/IWSLT/Data/test_speechre.tsv", "w", encoding="utf-8")as f:
    f.write(df.to_csv(sep="\t", index=False))