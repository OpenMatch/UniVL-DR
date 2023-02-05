export CUDA_VISIBLE_DEVICES=0

python gen_embeddings.py --out_path checkpoint_multi_inb \
--checkpoint checkpoint_multi_inb/model.best.pt \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs.json \
--query_path ../data/train.json \
--encode_img \
--encode_txt

python gen_embeddings.py --out_path checkpoint_multi_inb \
--checkpoint checkpoint_multi_inb/model.best.pt \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs.json \
--query_path ../data/train.json \
--encode_query

mv ./checkpoint_multi_inb/query_embedding.pkl ./checkpoint_multi_inb/train_query_embedding.pkl 

python gen_embeddings.py --out_path checkpoint_multi_inb \
--checkpoint checkpoint_multi_inb/model.best.pt \
--img_feat_path ../data/imgs.tsv \
--img_linelist_path ../data/imgs.lineidx.new \
--doc_path ../data/all_docs.json \
--cap_path ../data/all_imgs.json \
--query_path ../data/dev.json \
--encode_query

mv ./checkpoint_multi_inb/query_embedding.pkl ./checkpoint_multi_inb/dev_query_embedding.pkl 

python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb/train_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb/txt_embedding.pkl \
--data_path ../data/train.json \
--out_path ./checkpoint_multi_inb/train_all.json

python get_hard_negs_all.py   --query_embed_path ./checkpoint_multi_inb/dev_query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb/img_embedding.pkl \
--txt_embed_path ./checkpoint_multi_inb/txt_embedding.pkl \
--data_path ../data/dev.json \
--out_path ./checkpoint_multi_inb/dev_all.json

