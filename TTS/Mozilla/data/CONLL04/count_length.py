import pandas as pd
import json
import soundfile as sf
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']


df = pd.read_csv("conll04_train.tsv", sep="\t")
df.append(pd.read_csv("conll04_dev.tsv", sep="\t"))
df.append(pd.read_csv("conll04_test.tsv", sep="\t"))


# df = df.drop(df[df["duration_ms"] > 60000].index)
print("平均时长:", df["duration_ms"].mean().round(2), "ms")
print("平均帧数:", df["n_frames"].mean().round(2))


# sns.distplot(df["duration_ms"])
# plt.title("conll04_duration_ms")
# plt.show()