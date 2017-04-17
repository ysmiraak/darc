#!/bin/bash

ud2_treebank_path="/data/ud-treebanks-conll2017/"
udpipe_model_path="/data/udpipe-ud-2.0-conll17-170315/models/"
silver_train_path="./lab/silver_train/"
pretrain_w2v_path="./lab/pretrain_w2v/"

# ud2="ar bg ca cs cs_cac cs_cltt cu da de el en en_partut es es_ancora et
# eu fa fi fi_ftb fr fr_partut fr_sequoia ga gl gl_treegal got grc grc_proiel he hi hr
# hu it it_partut ja kk la la_ittb la_proiel lv nl nl_lassysmall no_bokmaal no_nynorsk pl
# pt ro ru ru_syntagrus sk sl sl_sst sv tr uk ur vi zh"

# ud2_no_lemma="en_lines id sv_lines ug pt_br ko"

ud2="kk"
ud2_no_lemma="ug"

##############
# run udpipe #
##############

for lang in ${ud2} ${ud2_no_lemma}
do
    model=${udpipe_model_path}${lang}".udpipe"
    infile=$(find ${ud2_treebank_path}*"/"${lang}"-ud-train.conllu")
    outfile=${silver_train_path}${lang}".conllu"
    udpipe --input conllu --tag --outfile ${outfile} ${model} ${infile}
    echo "written" ${silver_train_path}${lang}".conllu"
done

###############
# prepare raw #
###############

python darc_select.py -v \
       --data $(echo ${ud2} | sed -E "s|([^[:blank:]]+)|${silver_train_path}\1.conllu|g") \
       --form $(echo ${ud2} | sed -E "s|([^[:blank:]]+)|${pretrain_w2v_path}\1-form.raw|g") \
       --lemm $(echo ${ud2} | sed -E "s|([^[:blank:]]+)|${pretrain_w2v_path}\1-lemm.raw|g")

python darc_select.py -v \
       --data $(echo ${ud2_no_lemma} | sed -E "s|([^[:blank:]]+)|${silver_train_path}\1.conllu|g") \
       --form $(echo ${ud2_no_lemma} | sed -E "s|([^[:blank:]]+)|${pretrain_w2v_path}\1-form.raw|g")

################
# run word2vec #
################

for lang in ${ud2}
do
    word2vec -size 32 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${pretrain_w2v_path}${lang}"-form.raw" \
             -output ${pretrain_w2v_path}${lang}"-form.w2v" \
             -binary 1 -threads 4

    word2vec -size 32 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${pretrain_w2v_path}${lang}"-lemm.raw" \
             -output ${pretrain_w2v_path}${lang}"-lemm.w2v" \
             -binary 1 -threads 4
done

for lang in ${ud2_no_lemma}
do
    word2vec -size 64 \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${pretrain_w2v_path}${lang}"-form.raw" \
             -output ${pretrain_w2v_path}${lang}"-form.w2v" \
             -binary 1 -threads 4
done
