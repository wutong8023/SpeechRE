import json

with open("vocab.json", "r")as f:
    data = json.load(f)

with open("dict.txt", "w")as f:
    for k, v in data.items():
        f.write(k + " 1\n")