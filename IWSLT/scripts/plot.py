import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

a = [61.76,	61.53,	62.06,	64.18,	62.10]
b = [54.55,	52.48,	52.81,	58.60,	53.59]
c = [20.77,	27.33,	38.66,	44.19,	48.08]

df = pd.DataFrame([a, b, c], columns=["20%", "40%", "60%", "80%", "100%"])
# print(df)
# sns.lineplot(df, x=[range(5)],  y=["20%", "40%", "60%", "80%", "100%"])

plt.plot([20, 40, 60, 80, 100], a, label="Text+SpERT")
plt.plot([20, 40, 60, 80, 100], b, label="ASR+SpERT")
plt.plot([20, 40, 60, 80, 100], c, label="SpeechRE")
plt.xticks([20, 40, 60, 80, 100])
plt.legend()
plt.xlabel("Dataset completeness(%)")
plt.ylabel("F1 score(%)")
plt.grid()
plt.show()
