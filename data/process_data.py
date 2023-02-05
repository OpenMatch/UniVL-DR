import json
import numpy as np

np.random.seed(1234)


def process_example(example):
    img_negFacts = example['img_negFacts']
    img_negFacts = [str(img['image_id']) for img in img_negFacts]
    img_posFacts = example['img_posFacts']
    img_posFacts = [str(img['image_id']) for img in img_posFacts]
    txt_posFacts = example['txt_posFacts']
    txt_posFacts = [text['snippet_id'] for text in txt_posFacts]
    txt_negFacts = example['txt_negFacts']
    txt_negFacts = [text['snippet_id'] for text in txt_negFacts]
    new_example = {'qid': example['qid'], 'Q': example['Q'].strip().strip('"').replace('\r', '').strip(), 'A': example['A'], 'img_posFacts': img_posFacts,
                   'img_negFacts': img_negFacts, 'txt_negFacts': txt_negFacts, 'txt_posFacts': txt_posFacts}
    return new_example


def get_qrels(example):
    qrel_list = []
    qid = example['qid']
    img_posFacts = example['img_posFacts']
    txt_posFacts = example['txt_posFacts']
    for imgid in img_posFacts:
        qrel_list.append([qid, imgid, '1'])
    for did in txt_posFacts:
        qrel_list.append([qid, did, '1'])

    return qrel_list


docs = {}
images = {}
train_data = []
validate_data = []
with open('./WebQA_train_val.json') as fin:
    data = json.load(fin)
    for qid, example in data.items():
        example['qid'] = qid
        if example['split'] == 'train':
            train_data.append(process_example(example))
        else:
            validate_data.append(process_example(example))
        neg_facts = example['txt_negFacts']
        pos_facts = example['txt_posFacts']
        for fact in pos_facts + neg_facts:
            did = fact['snippet_id']
            fact['fact'] = fact['fact'].strip().strip('"').replace('\r', '').strip()
            docs[did] = fact
        pos_facts = example['img_posFacts']
        neg_facts = example['img_negFacts']
        for img in pos_facts + neg_facts:
            imgid = img['image_id']
            img = {'title': img['title'], 'caption': img['caption'].strip().strip('"').replace('\r', '').strip(), 'image_id': img['image_id']}
            images[imgid] = img

with open('./WebQA_test.json') as fin:
    data = json.load(fin)
    for qid, example in data.items():
        example['qid'] = qid
        facts = example['txt_Facts']
        for fact in facts:
            did = fact['snippet_id']
            fact['fact'] = fact['fact'].strip().strip('"').replace('\r', '').strip()
            docs[did] = fact
        facts = example['img_Facts']
        for img in facts:
            imgid = img['image_id']
            img = {'title': img['title'], 'caption': img['caption'].strip().strip('"').replace('\r', '').strip(), 'image_id': img['image_id']}
            images[imgid] = img


print (len(docs))
print (len(images))
np.random.shuffle(train_data)
print (len(train_data))
print (len(validate_data))

with open('all_docs.json', 'w') as fout:
    for did, doc in docs.items():
        fout.write(json.dumps(doc) + '\n')

with open('all_imgs.json', 'w') as fout:
    for did, image in images.items():
        fout.write(json.dumps(image) + '\n')

dev_qrels = []
test_qrels = []
with open('train.json', 'w') as fout, open('dev.json', 'w') as fout1:
    counter = 0
    for step, example in enumerate(train_data):
        if len(example['img_posFacts']) != 0 and len(example['txt_posFacts']) != 0:
            counter += 1
        if step < 5000:
            fout1.write(json.dumps(example) + '\n')
            dev_qrels.extend(get_qrels(example))
        else:
            fout.write(json.dumps(example) + '\n')
print (counter)
with open('test.json', 'w') as fout:
    counter = 0
    for example in validate_data:
        fout.write(json.dumps(example) + '\n')
        if len(example['img_posFacts']) != 0 and len(example['txt_posFacts']) != 0:
            counter += 1
        test_qrels.extend(get_qrels(example))
print (counter)
with open("dev_qrels.txt", "w") as fout:
    for example in dev_qrels:
        fout.write("\t".join(example) + "\n")
with open("test_qrels.txt", "w") as fout:
    for example in test_qrels:
        fout.write("\t".join(example) + "\n")



