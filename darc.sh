train="/data/ud-treebanks-conll2017/UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu"
lang="grc_proiel"

embed="./lab/embed/"

python darc_extract.py -v \
       --data $train \
       --form $embed$lang".form.raw" \
       --lemm $embed$lang".lemm.raw"
