import sys
import csv
from tqdm import tqdm
import collections
import gzip
import pickle
import faiss
import os
import logging
import argparse
import json
import os.path as op
import numpy as np
logger = logging.getLogger()
import random
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference



def load_file(path):
    all_data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            all_data[example['qid']] = example
    return all_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser("")
    parser.add_argument("--query_embed_path")
    parser.add_argument("--txt_embed_path")
    parser.add_argument("--img_embed_path")
    parser.add_argument("--data_path", default='../data/test.json')
    parser.add_argument("--out_path")

    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--topN", type=int, default=100)
    parser.add_argument("--max_neg", type=int, default=100)


    args = parser.parse_args()


    handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logger.info(args)
    all_idx = []
    all_embeds = []
    faiss.omp_set_num_threads(16)
    cpu_index = faiss.IndexFlatIP(args.dim)


    with open(args.query_embed_path, 'rb') as fin:
        logger.info("load data from {}".format(args.query_embed_path))
        query_idx, query_embeds = pickle.load(fin)
        query_embeds = np.array(query_embeds, np.float32)
    data_dict = load_file(args.data_path)
    if args.txt_embed_path:
        logger.info("load data from {}".format(args.txt_embed_path))
        with open(args.txt_embed_path, 'rb') as fin:
            txt_idx, txt_embeds = pickle.load(fin)
        cpu_index.add(np.array(txt_embeds, np.float32))
        all_idx.extend(txt_idx)

    if args.img_embed_path:
        logger.info("load data from {}".format(args.img_embed_path))
        with open(args.img_embed_path, 'rb') as fin:
            img_idx, img_embeds = pickle.load(fin)
        cpu_index.add(np.array(img_embeds, np.float32))
        all_idx.extend(img_idx)

    D, I = cpu_index.search(query_embeds, args.topN)
    txt_idx_dict = {}
    for idx in txt_idx:
        txt_idx_dict[idx] = 1
    assert len(query_idx) == len(I)

    for step, qid in enumerate(query_idx):
        instance = data_dict[qid]
        pos_ids = set(instance['txt_posFacts'] + instance['img_posFacts'])
        neg_ids = []
        neg_txt_ids = []
        neg_img_ids = []
        for idx in I[step]:
            real_idx = all_idx[idx]
            if real_idx not in pos_ids:
                neg_ids.append(real_idx)
                if real_idx in txt_idx_dict:
                    neg_txt_ids.append(real_idx)
                else:
                    neg_img_ids.append(real_idx)
            if len(neg_ids) > args.max_neg:
                break
        data_dict[qid]['all_negFacts'] = neg_ids
        data_dict[qid]['img_negFacts'] = neg_img_ids
        data_dict[qid]['txt_negFacts'] = neg_txt_ids
    del cpu_index

    logger.info("Save file!")
    with open(args.out_path, "w") as fout:
        for _, instance in data_dict.items():
            fout.write(json.dumps(instance) + "\n")


