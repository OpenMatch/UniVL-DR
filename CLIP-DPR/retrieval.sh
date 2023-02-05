python retrieval.py   --query_embed_path  ./checkpoint_multi_inb/query_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb/img_embedding.pkl \
--qrel_path ../data/test_qrels.txt

python retrieval.py   --query_embed_path  ./checkpoint_multi_inb/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_inb/txt_embedding.pkl \
--img_embed_path ./checkpoint_multi_inb/img_embedding.pkl \
--qrel_path ../data/test_qrels.txt


python retrieval.py   --query_embed_path  ./checkpoint_multi_inb/query_embedding.pkl \
--doc_embed_path ./checkpoint_multi_inb/txt_embedding.pkl \
--qrel_path ../data/test_qrels.txt
