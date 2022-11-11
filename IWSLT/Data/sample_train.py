import random
import json
import pandas as pd
from tqdm import tqdm


dataset = "dev"

df = pd.read_csv(dataset + "_tacred_top5.tsv", sep="\t")

print(len(df["id"]))

random.seed(2022)
sample_list80 = random.sample(range(len(df["id"])), int(len(df["id"])*0.8))
sample_list60 = random.sample(range(len(sample_list80)), int(len(sample_list80)*0.75))
sample_list40 = random.sample(range(len(sample_list60)), int(len(sample_list60)*0.66))
sample_list20 = random.sample(range(len(sample_list40)), int(len(sample_list40)*0.5))

with open(dataset + "_tacred_top5_80.tsv", "w", encoding="utf-8")as f:
    df_new = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
    for i in tqdm(sample_list80):
        df_new.loc[len(df_new["id"])] = df.loc[i]
    f.write(df_new.to_csv(sep="\t", index=False))

with open(dataset + "_tacred_top5_60.tsv", "w", encoding="utf-8")as f:
    df_new = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
    for i in tqdm(sample_list60):
        df_new.loc[len(df_new["id"])] = df.loc[i]
    f.write(df_new.to_csv(sep="\t", index=False))

with open(dataset + "_tacred_top5_40.tsv", "w", encoding="utf-8")as f:
    df_new = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
    for i in tqdm(sample_list40):
        df_new.loc[len(df_new["id"])] = df.loc[i]
    f.write(df_new.to_csv(sep="\t", index=False))

with open(dataset + "_tacred_top5_20.tsv", "w", encoding="utf-8")as f:
    df_new = pd.DataFrame(columns=["id", "audio", "duration_ms", "n_frames", "tgt_text", "speaker", "tgt_lang"])
    for i in tqdm(sample_list20):
        df_new.loc[len(df_new["id"])] = df.loc[i]
    f.write(df_new.to_csv(sep="\t", index=False))