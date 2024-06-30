#  Universal Vision-Language Dense Retrieval (UniVL-DR)
There are source codes for Universal Vision-Language Dense Retrieval [Our Paper](https://openreview.net/pdf?id=PQOlkgsBsik).


## Requirement
* Python==3.7
* Pytorch
* transformers
* clip
* faiss-cpu==1.7.0
* tqdm
* numpy
* base64
* Install the ``pytrec_eval`` from ``https://github.com/cvangysel/pytrec_eval``


## Data and Checkpoint
* All these files can be downloaded and you should put them in the corresponding folders.
* All ``data`` can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/data.zip). Please note that the ``imgs.tsv`` file should be downloaded from the project of WebQA (by downloading the data from [this link](https://drive.google.com/drive/folders/1ApfD-RzvJ79b-sLeBx1OaiPNUYauZdAZ?usp=sharing) and running ```7z x imgs.7z.001```).
* The ``checkpoint_multi_inb`` (The checkpoint of CLIP-DPR) can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/checkpoint_multi_inb.zip).
* The ``checkpoint_multi_hn``  (The checkpoint of UniVL-DR) can be found at [Ali Drive](https://thunlp.oss-cn-qingdao.aliyuncs.com/UniVLDR/checkpoint_multi_hn.zip).

## Train UniVL-DR
* UniVL-DR inherits CLIP (ViT-B/32). The texts must be truncated by 77 tokens and you can try different vision-language models. As shown in our experiments, we suggest to use the dual encoder models.
* There are two steps to train UniVL-DRR:
* First step: Go to the ``CLIP-DPR`` folder and train models using inbatch negatives:
```
bash train_multi.sh
```
* Second step: Then using CLIP-DPR to generate hard negatives for training UniVL-DR: 
```
bash get_hn.sh
```
* Final step: Go to the ``UniVL-DR`` folder and train models using hard negatives: 
```
bash train_multi.sh
```

## Evaluate Retrieval Effectiveness
* These experimental results are shown in Table 1 of our paper.
* Go to the ``CLIP-DPR`` or ``UniVL-DR`` folder and evaluate model performance as follow:
```
bash gen_embeds.sh
```
```
bash retrieval.sh
```



## Results
The results are shown as follows.
| Setting             | Model                               | MRR@10 | NDCG@10 | MRR@20 | NDCG@20 | Rec@20 | Rec@100 |
|------------------------------|----------------------------------------------|:---------------:|:----------------:|:---------------:|:----------------:|:---------------:|:----------------:|
| Single Modality\\(Text Only) | BM25                                         |      53.75      |       49.60      |      54.10      |       51.72      |      68.16      |       80.69      |
|                              | DPR (Zero-Shot)   |      22.72      |       20.06      |      23.14      |       21.79      |      32.78      |       45.43      |
|                              | CLIP (Zero-Shot) |      18.16      |       16.76      |      18.60      |       18.27      |      27.97      |       39.83      |
|                              | BERT-DPR          |      42.16      |       39.57      |      42.76      |       42.26      |      60.85      |       77.10      |
|                              | NQ-DPR            |      41.88      |       39.65      |      42.44      |       42.35      |      61.71      |       78.57      |
|                              | NQ-ANCE         |      45.54      |       42.05      |      45.93      |       43.83      |      58.42      |       69.31      |
| Divide-Conquer               | VinVL-DPR                                    |      22.11      |       22.92      |      22.80      |       25.41      |      46.27      |       62.82      |
|                              | CLIP-DPR                                     |      37.35      |       37.56      |      37.93      |       40.77      |      69.38      |       85.53      |
|                              | BM25 & CLIP-DPR                             |      42.27      |       41.58      |      42.79      |       44.69      |      73.34      |       87.50      |
|                              | BM25 & CLIP-DPR (Oracle Modality)           |      61.05      |       58.18      |      61.37      |       60.45      |  80.82 |  90.83  |
| UnivSearch                   | CLIP (Zero-Shot)                             |      10.59      |       8.69       |      10.80      |       9.52       |      14.32      |       20.21      |
|                              | VinVL-DPR                                    |      38.14      |       35.43      |      38.74      |       37.79      |      53.89      |       69.42      |
|                              | CLIP-DPR                                     |      48.83      |       46.32      |      49.34      |       49.11      |      69.84      |       86.43      |
|                              | UniVL-DR                                     |  62.40 |  59.32  |  62.69 |  61.22  |      80.37      |       89.42      |






## Citation
```
@inproceedings{liu2023univldr,
  title={Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval},
  author={Liu, Zhenghao and Xiong, Chenyan and Lv, Yuanhuiyi and Liu, Zhiyuan and Yu, Ge},
  booktitle={Proceedings of ICLR},
  year={2023}
}
```

## Contact
If you have questions, suggestions, and bug reports, please email:
```
liuzhenghao0819@gmail.com
```
