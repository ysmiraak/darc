train="./thesis/train/"
embed="./thesis/embed/"
langs="ar bg eu fa fi_ftb grc he hr it la_proiel nl pl sv tr zh"

for lang in ${langs}
do
    python darc_select.py -v \
           --data ${train}${lang}"-ud-dev.conllu" \
           --form ${embed}${lang}"-form.raw" \
           --lemm ${embed}${lang}"-lemm.raw"

    word2vec -size 64 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${embed}${lang}"-form.raw" \
             -output ${embed}${lang}"-form64.w2v" \
             -binary 1

    word2vec -size 32 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${embed}${lang}"-form.raw" \
             -output ${embed}${lang}"-form32.w2v" \
             -binary 1

    word2vec -size 32 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${embed}${lang}"-lemm.raw" \
             -output ${embed}${lang}"-lemm32.w2v" \
             -binary 1
done
