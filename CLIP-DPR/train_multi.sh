export CUDA_VISIBLE_DEVICES=0

python train.py  --out_path ./checkpoint_multi_inb/ \
--train_path ../data/train.json \
--valid_path ../data/dev.json \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs.json \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new