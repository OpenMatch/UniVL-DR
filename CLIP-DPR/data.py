import json
import os
from visual import TSVFile
import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WebQADataset(Dataset):
    def __init__(self, args, preprocess_fn, data, docs, captions, shuffle):
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn

        self.img_map = {}
        self.img_tsv = []
        self.docs = docs
        self.captions = captions

        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)
        self.data = data


    def __len__(self):
        return len(self.data)


    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(Image.open(io.BytesIO(base64.b64decode(img))))
        if self.captions != None:
            cap = self.captions[idx]
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        txt_labels = []
        img_labels = []
        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            if 'pos_img' in example:
                img_inputs.append(example['pos_img']['img'])
                if 'cap' in example['pos_img']:
                    cap_inputs.append(example['pos_img']['cap'])
                img_labels.append(qid)
            if 'pos_txt' in example:
                txt_inputs.append(example['pos_txt'])
                txt_labels.append(qid)
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    img_inputs.append(instance['img'])
                    if 'cap' in instance:
                        cap_inputs.append(instance['cap'])
                    img_labels.append(-1)
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    txt_inputs.append(instance)
                    txt_labels.append(-1)

        processed_batch['queries'] = clip.tokenize(queries, truncate=True)
        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
            processed_batch['img_labels'] = img_labels
            if len(cap_inputs) != 0:
                assert len(cap_inputs) == len(img_inputs)
                processed_batch['img_caps'] = clip.tokenize(cap_inputs, truncate=True)

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = clip.tokenize(txt_inputs, truncate=True)
            processed_batch['txt_labels'] = txt_labels

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['Q']
        instance = {'query': query}

        if len(example['img_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['img_posFacts'])
            else:
                idx = example['img_posFacts'][0]
            instance["pos_img"] = self.encode_img(idx)
        elif len(example['txt_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['txt_posFacts'])
            else:
                idx = example['txt_posFacts'][0]
            instance["pos_txt"] = self.docs[idx]
        else:
            raise ('No positive instance!')



        if self.img_neg_num > 0:
            neg_imgs = []
            neg_img_idx = example['img_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_img_idx)
            neg_img_idx = neg_img_idx[:self.img_neg_num]
            for idx in neg_img_idx:
                neg_imgs.append(self.encode_img(idx))
            instance["neg_imgs"] = neg_imgs

        if self.txt_neg_num > 0:
            neg_txts = []
            neg_txt_idx = example['txt_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_txt_idx)
            neg_txt_idx = neg_txt_idx[:self.txt_neg_num]
            for idx in neg_txt_idx:
                neg_txts.append(self.docs[idx])
            instance["neg_txts"] = neg_txts
        return instance




def load_file(path, txt=True, img=True):
    data = []
    with open(path) as fin:
        assert (txt or img)
        for line in fin:
            example = json.loads(line.strip())
            txt_negFacts = example['txt_negFacts']
            np.random.shuffle(txt_negFacts)
            example['txt_negFacts'] = txt_negFacts

            img_negFacts = example['img_negFacts']
            np.random.shuffle(img_negFacts)
            example['img_negFacts'] = img_negFacts

            if txt and len(example['txt_posFacts']) != 0:
                data.append(example)
            if img and len(example['img_posFacts']) != 0:
                data.append(example)
    return data

def load_docs(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            data[did] = example['fact']
    return data

def load_caps(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            data[imgid] = example['caption']
    return data

