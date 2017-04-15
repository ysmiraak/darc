python darc-extract.py \
       -v \
       --data \
       /data/ud-treebanks-conll2017/UD_Hebrew/he-ud-train.conllu \
       /data/ud-treebanks-conll2017/UD_Latin-PROIEL/la_proiel-ud-train.conllu \
       --form \
       ./embeddings/he.form.raw \
       ./embeddings/la_proiel.form.raw \
       --lemm \
       ./embeddings/he.lemm.raw \
       ./embeddings/la_proiel.lemm.raw 
