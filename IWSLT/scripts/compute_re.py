import torch
target = torch.tensor([[50265, 40825,  1250,  5374, 13275,   824, 50266, 38831,  1872,  1352,
         50267, 47024,  1215,  1121, 50266,  7083,   102, 50267, 47024,  1215,
          1121, 50265, 38831,  1872,  1352, 50266,  7083,   102, 50267, 47024,
          1215,  1121,     2],
        [50265, 27759,  2825,  1792, 16741,  5722,  3658, 50266,   863,   677,
         28721, 50267, 46532, 20930,  1215,  1121,     2,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1],
        [50265, 23565,  4127, 50266, 37211,  7436, 50267, 47024,  1215,  1121,
         50266,   487,     4,   975,     4, 50267, 47024,  1215,  1121, 50265,
         37211,  7436, 50266,   487,     4,   975,     4, 50267, 47024,  1215,
          1121,     2,     1],
        [50265,   387, 26787, 18488, 32371, 50266,  2068,  2747,    12,   565,
         17042, 50267, 21461,  1215,  2709,     2,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1],
        [50265, 43543,   717,  3411, 38054,   100, 35205,    38,   272,  2688,
           500, 38054,   100, 35205, 50266, 41384, 50267, 46532, 20930,  1215,
          1121,     2,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1],
        [50265, 34792,   808, 11737,  2154,  7822,  1417, 50266,  2068,  2747,
            12,   565, 17042, 50267, 21461,  1215,  2709,     2,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1],
        [50265,   717, 21851, 11434,   953, 50266,   448,  6502,  2716,  3355,
         50267, 46532, 20930,  1215,  1121,     2,     1,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1],
        [50265,   448, 48017,   264, 26390, 27322, 50266,  2068,  2747,    12,
           565, 17042, 50267, 21461,  1215,  2709,     2,     1,     1,     1,
             1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
             1,     1,     1]])
output =torch.tensor([[    0,     4,  1250,     4,     0,     0,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0,     0,     0,     0,   578,
             0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
             0,     0,     0],
        [    0,     4,     4,     4,  3572,     4,     6,     6,     8,     8,
             8,     0,     8,    91,     8,     8,     8,     8,     8,     8,
             8,     8,     8,     8,     8,     6,     8,     8,     6,     8,
             8,     8,     8],
        [    0,   895,     7,    84,    84,    84,    84,     0,     0,     0,
             0,    84,     0,     0,    84,     0,     0,     0,    84,     0,
             0,     0,     0,     0,     0,     0,     0,    84,     0,     0,
             0,    84,    84],
        [    0,  4484,     8,    84,    84,    84,    39,    17,     0,     0,
             0,    17,     7,   260,     0,     0,    17,     0,   832,     8,
             8,    44,    84,     0,     0,     0,     0,     8,    17,    84,
             6,    20,    84],
        [    0,     6,  2118,  3850,     6,     6,     6,     6,     6,     7,
             6,     6,     6,     6,    17,     6,     7,     6,     6,     6,
             6,     0,     6,     6,     6,     6,     0,     7,     6,     6,
             6,     6,     6],
        [    0,   176,   885,     8, 14887,    20,    52, 20583,   286,    20,
             8,   166,     8,    52,    20,   166,   166,    38,   449,   166,
            13,    20,    52,    20,   166,    52,    20,   170,    52,    13,
             0, 35382,    52],
        [    0,     4,    22,     6,   119,     4,   108,   108,     8,   108,
           108,   108,  6785,   108,  6785,  6785,  6785,  6785,    29,  6785,
          6785,  6785,  6785,     6,   108,  6785,  1708,  6785,   133,  6785,
          6785,   133,  6785],
        [    0,     4,  1589,  1589,  1589,   170,  1589,  1589,  1589,  1589,
          1589,  1589,  1589,  1589,  1589,   155,  1589,  1589,  1589,  1589,
          1589,  1589,     0,  1589,  1589,     0,  1589,  1589,  1589,     0,
             0,  1589,  1589]])


def extract_triplets(text):
    del_list = [0, 1, 2]
    text = [None if i in del_list else i for i in text]
    text = str(text).replace(",", "").replace("None", "").strip("[ ]")
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    current = 'x'
    for token in text.split():
        if token == "50265":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "50266":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "50267":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    return triplets


assert target.shape == output.shape
p = r = 0
total_p = total_r = 1e-5
for i in range(target.shape[0]):
    target_triplets = extract_triplets(target[i].tolist())
    output_triplets = extract_triplets(output[i].tolist())
    target_relations = []
    output_relations = []
    for i in target_triplets:
        target_relations.append(tuple(i.values()))
    for i in output_triplets:
        output_relations.append(tuple(i.values()))
    for i in target_relations:
        if i in output_relations:
            r += 1
        total_r += 1
    for i in output_relations:
        if i in target_relations:
            p += 1
        total_p += 1
    print(target_relations)
p = p / total_p
r = r / total_r
f1 = 2 * p * r / (p + r + 1e-5)
total = target.shape[0]

print(p, r, f1, total)