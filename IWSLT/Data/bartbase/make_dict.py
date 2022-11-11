import json

with open("/data/wangguitao/IWSLT/Pre-trained_models/BART-base/dict.txt")as f:
    data0 = f.readlines()

with open("vocab.json", "r")as f:
    data = json.load(f)
    for line in data0:
        if line.startswith("madeup"):
            line = line.strip().split()[0]
            data[line] = len(data)-1

with open("vocab2.json", "w", encoding="utf-8")as f:
    json.dump(data, f, ensure_ascii=False)

with open("dict.txt", "w")as f:
    for k, v in data.items():
        f.write(k + " 1\n")