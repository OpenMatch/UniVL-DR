import json
from visual import TSVFile
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import base64
from PIL import Image
import io
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import clip
from data import load_caps, load_docs, load_file, WebQADataset
logger = logging.getLogger()
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

class ImgDataset(Dataset):
    def __init__(self, args, preprocess, captions=None):
        self.max_seq_len = args.max_seq_len

        self.img_map = {}
        self.img_ids = []
        self.captions = captions
        self.preprocess_fn = preprocess

        all_img_num = 0
        with open(args.img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
                self.img_ids.append(tokens[0])
        self.img_tsv = TSVFile(args.img_feat_path, all_img_num)

    def __len__(self):
        return len(self.img_ids)

    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(Image.open(io.BytesIO(base64.b64decode(img))))
        if self.captions != None:
            cap = self.captions[idx]
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        img_inputs = []
        img_caps = []
        idx_list = []

        for example in batch:
            img_inputs.append(example['img_inputs'])
            if 'img_caps' in example:
                img_caps.append(example['img_caps'])
            idx_list.append(example['idx'])
        processed_batch = {}
        processed_batch['idx_list'] = idx_list
        processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
        if len(img_caps) != 0:
            processed_batch['img_caps'] = clip.tokenize(img_caps, truncate=True)
        return processed_batch

    def __getitem__(self, index):
        img_idx = self.img_ids[index]
        img_inputs = self.encode_img(img_idx)
        instance = {
            'idx': img_idx,
            'img_inputs': img_inputs['img']
        }
        if 'cap' in img_inputs:
            instance['img_caps'] = img_inputs['cap']

        return instance

class TextDataset(Dataset):
    def __init__(self, data, max_len):
        self.max_len = max_len
        self.data = data

    def __len__(self):
        return len(self.data)




    def Collector(self, batch):
        txt_inputs = []
        idx_list = []

        for qid, example in enumerate(batch):
            txt_inputs.append(example['txt_inputs'])
            idx_list.append(example['idx'])
        processed_batch = {
            'txt_inputs': clip.tokenize(txt_inputs, truncate=True),
            'idx_list': idx_list
        }
        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        txt_inputs = example[1]

        return {
            'idx': example[0],
            'txt_inputs': txt_inputs
        }




def gen_embeddings(model, valid_reader, outpath):
    model.eval()
    all_embeddings = []
    all_index = []
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            idx_list = batch['idx_list']
            if 'img_inputs' in batch:
                embeddings = model.encode_image(batch['img_inputs'].cuda())
                if 'img_caps'  in batch:
                    cap_embeddings = model.encode_text(batch['img_caps'].cuda())
                    embeddings = embeddings + cap_embeddings

            else:
                embeddings = model.encode_text(batch['txt_inputs'].cuda())
            embeddings = F.normalize(embeddings, dim=-1)
            embeddings = embeddings.cpu()
            assert len(embeddings) == len(idx_list)
            all_index.extend(idx_list)
            all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    with open(outpath, 'wb') as fout:
        pickle.dump((all_index, all_embeddings), fout)

def load_docs(path):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            data.append([did, example['fact']])
    return data


def load_queries(path):
    data = []
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            qid = str(example['qid'])
            data.append([qid, example['Q']])
    return data


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")
    parser.add_argument("--max_seq_len", type=int, default=77)

    parser.add_argument("--out_path", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--img_feat_path", type=str)
    parser.add_argument("--img_linelist_path", type=str)

    parser.add_argument("--query_path", type=str)
    parser.add_argument("--doc_path", type=str)
    parser.add_argument("--cap_path", type=str)

    parser.add_argument('--encode_txt', action='store_true', default=False)
    parser.add_argument('--encode_img', action='store_true', default=False)
    parser.add_argument('--encode_query', action='store_true', default=False)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()


    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.cuda()

    docs = load_docs(args.doc_path)
    if args.encode_query:
        queries = load_queries(args.query_path)
        query_data = TextDataset(queries, args.max_seq_len)
        query_sampler = SequentialSampler(query_data)
        query_reader = DataLoader(dataset=query_data, sampler=query_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=query_data.Collector)

        output = os.path.join(args.out_path, 'query_embedding.pkl')
        gen_embeddings(model, query_reader, output)

    if args.encode_img:
        captions = None
        if args.cap_path:
            captions = load_caps(args.cap_path)
        img_data = ImgDataset(args, preprocess, captions=captions)
        sampler = SequentialSampler(img_data)
        img_reader = DataLoader(dataset=img_data, sampler=sampler, num_workers=args.num_workers,
                                      batch_size=args.batch_size, collate_fn=img_data.Collector)
        output = os.path.join(args.out_path, 'img_embedding.pkl')
        gen_embeddings(model, img_reader, output)

    if args.encode_txt:
        docs = load_docs(args.doc_path)
        txt_data = TextDataset(docs, args.max_seq_len)
        txt_sampler = SequentialSampler(txt_data)
        txt_reader = DataLoader(dataset=txt_data, sampler=txt_sampler, num_workers=args.num_workers,
                                    batch_size=args.batch_size, collate_fn=txt_data.Collector)

        output = os.path.join(args.out_path, 'txt_embedding.pkl')
        gen_embeddings(model, txt_reader, output)