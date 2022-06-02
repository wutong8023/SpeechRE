import json
from tqdm import tqdm

for dataset in ["train", "dev", "test"]:
    print("dataset:", dataset)
    with open("conll04_" + dataset + "_merge.json", encoding="utf-8")as f:
        lines = json.load(f)
        print(dataset + ":", len(lines))

    with open("conll04_" + dataset + "_linear.json", "w", encoding="utf-8")as f:
        data_new = []
        for line in tqdm(lines):
            data = line
            items = data["linearization"]
            items.sort(key=lambda x: x[3], reverse=False)
            triplets = ''
            prev_head = None
            for item in items:
                if prev_head == item[0]:
                    triplets += ' <subj> ' + item[1] + ' <obj> ' + item[2]
                elif prev_head == None:
                    triplets += '<triplet> ' + item[0] + ' <subj> ' + item[1] + ' <obj> ' + item[2]
                    prev_head = item[0]
                else:
                    triplets += ' <triplet> ' + item[0] + ' <subj> ' + item[1] + ' <obj> ' + item[2]
                    prev_head = item[0]
            data_new.append({"id": data["id"], "text": data["text"], "linearization": triplets})
        json.dump(data_new, f, ensure_ascii=False, indent=2)