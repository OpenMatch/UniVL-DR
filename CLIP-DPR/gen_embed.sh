export CUDA_VISIBLE_DEVICES=0

python gen_embeddings.py --out_path checkpoint_multi_inb \
--checkpoint checkpoint_multi_inb/model.best.pt \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs.json \
--query_path ../data/test.json \
--encode_query \
--encode_img \
--encode_txt
