import json
from tqdm import tqdm


for dataset in ["train", "dev", "test"]:
    print("dataset:", dataset)
    with open("conll04_" + dataset + ".json", encoding="utf-8")as f:
        lines = json.load(f)
        print(dataset + ":", len(lines))

    data_new = []
    for id, line in tqdm(enumerate(lines)):
        data = line
        text = " ".join(data["tokens"])
        entities = data["entities"]
        relations = data["relations"]
        save_list = []
        for relation in relations:
            head = " ".join(data["tokens"][entities[relation["head"]]["start"]: entities[relation["head"]]["end"]])
            pos = entities[relation["head"]]["start"]
            tail = " ".join(data["tokens"][entities[relation["tail"]]["start"]: entities[relation["tail"]]["end"]])
            save = [head, tail, relation["type"], pos]
            save_list.append(save)
        data_new.append({"id": dataset + "-" + str(id), "text": text, "linearization": save_list})


    print(dataset + ":", len(data_new))

    with open("conll04_" + dataset + "_merge.json", "w", encoding="utf-8")as f:
        json.dump(data_new, f, ensure_ascii=False, indent=2)
