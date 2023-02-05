export CUDA_VISIBLE_DEVICES=0
 python train.py  --out_path ./checkpoint_multi_hn/ \
--train_path ../CLIP-DPR/checkpoint_multi_inb/train_all.json \
--valid_path ../CLIP-DPR/checkpoint_multi_inb/dev_all.json \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs_query.json \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new \
--train_batch_size 64 \
--valid_batch_size 64 \
--pretrained_model_path ../CLIP-DPR/checkpoint_multi_inb/model.best.pt \
--gradient_accumulation_steps 1 \
--img_neg_num 1 \
--txt_neg_num 1
