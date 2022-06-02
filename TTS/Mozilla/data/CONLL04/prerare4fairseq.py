import pandas as pd
import json
import soundfile as sf
from tqdm import tqdm


for dataset in ["train", "dev", "test"]:
    df = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
    with open("conll04_" + dataset + "_linear.json", "r", encoding="utf-8")as f:
        lines = json.load(f)
        for id, line in enumerate(tqdm(lines)):
            file_name = "/data/wangguitao/IWSLT/Datasets/conll04/" + dataset + "/conll04_" + dataset + "-" + str(id) + "_16k.wav"
            audio_input, sample_rate = sf.read(file_name)
            assert sample_rate == 16000
            df.loc[id] = [line["id"],
                          file_name + ":0:" + str(len(audio_input)),
                          int(len(audio_input)/16),
                          len(audio_input),
                          line["linearization"],
                          "0",
                          "en"
                          ]
    with open(dataset + "_conll04.tsv", "w", encoding="utf-8")as f:
        f.write(df.to_csv(sep="\t", index=False))
