train="/data/ud-treebanks-conll2017/UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu"
lang="grc_proiel"

embed="./0000/embed/"

python darc-extract.py -v \
       --data $train \
       --form $embed$lang".form.raw" \
       --lemm $embed$lang".lemm.raw"
