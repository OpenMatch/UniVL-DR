import argparse
import os
import logging
import base64
import os.path as op
import random, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

class TSVFile(object):
    def __init__(self, tsv_file, total_num):
        self.tsv_file = tsv_file
        self._fp = None
        self.pid = None
        self.total_num = total_num

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        return self.total_num

    def seek(self, pos):
        self._ensure_tsv_opened()
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def __getitem__(self, pos):
        return self.seek(pos)

    def __len__(self):
        return self.num_rows()


    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

