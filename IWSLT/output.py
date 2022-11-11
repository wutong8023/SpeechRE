def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.replace("<triplet>", " <triplet> ").replace("<subj>", " <subj> ").replace("<obj>", " <obj> ").strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
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


with open("/data/wangguitao/IWSLT/Save/speechre_tacred_top5_part_part/results/generate-test_speechre.txt")as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("D"):
            linear = line.split("\t")[-1].strip()
            print("Linearization triplet:", linear)
            print("Relation triplet(s):", extract_triplets(linear))

'''
argon2:$argon2id$v=19$m=10240,t=10,p=8$6aA3le97f/BMa7w7++/yEQ$jxOTRgfHd48bTTJF+d3hygLSetrh8qlnvDas45ChuLo
'''
