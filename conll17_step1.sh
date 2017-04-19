#!/bin/bash

# ud2="ar bg ca cs cs_cac cs_cltt cu da de el en en_lines en_partut es es_ancora et
#  eu fa fi fi_ftb fr fr_partut fr_sequoia ga gl gl_treegal got grc grc_proiel he hi hr
#  hu id it it_partut ja kk ko la la_ittb la_proiel lv nl nl_lassysmall no_bokmaal no_nynorsk pl
#  pt pt_br ro ru ru_syntagrus sk sl sl_sst sv sv_lines tr ug uk ur vi zh"

ud2_treebank_path="/data/ud-treebanks-conll2017/"
udpipe_model_path="./lab/udpipe_model/"
silver_train_path="./lab/silver_train/"
pretrain_w2v_path="./lab/pretrain_w2v/"
udpiped_test_path="./lab/udpiped_test/"

# hi hr
# hu id it it_partut ja kk ko la la_ittb la_proiel lv nl nl_lassysmall no_bokmaal no_nynorsk pl
# pt pt_br ro ru ru_syntagrus sk sl sl_sst sv sv_lines tr ug uk ur vi zh

ud2=

##############
# run udpipe #
##############

for lang in ${ud2}
do
    model=${udpipe_model_path}${lang}".udpipe"
    # preparing training data
    gold=$(find ${ud2_treebank_path}*"/"${lang}"-ud-train.conllu")
    silver=${silver_train_path}${lang}".conllu"
    udpipe --input conllu --tag --outfile ${silver} ${model} ${gold}
    echo "written" ${silver}
    # preparing testing data
    raw=$(find ${ud2_treebank_path}*"/"${lang}"-ud-dev.txt" ||
              find ${ud2_treebank_path}*"/"${lang}"-ud-train.txt")
    piped=${udpiped_test_path}${lang}".conllu"
    udpipe --input horizontal --tokenize --tag --outfile ${piped} ${model} ${raw}
    echo "written" ${piped}
done

###############
# prepare raw #
###############

for lang in ${ud2}
do
    case ${lang} in
        "en_lines"|"id"|"ko"|"pt_br"|"sv_lines"|"ug")
            python darc_select.py -v \
                   --data ${silver_train_path}${lang}".conllu" \
                   --form ${pretrain_w2v_path}${lang}"-form.raw"
          ;;
        *)
            python darc_select.py -v \
                   --data ${silver_train_path}${lang}".conllu" \
                   --form ${pretrain_w2v_path}${lang}"-form.raw" \
                   --lemm ${pretrain_w2v_path}${lang}"-lemm.raw"
            ;;
    esac
done

################
# run word2vec #
################

for lang in ${ud2}
do
    case ${lang} in
        "en_lines"|"id"|"ko"|"pt_br"|"sv_lines"|"ug")
            form_embed_dim=64
            ;;
        *)
            word2vec -size 32 \
                     -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
                     -train ${pretrain_w2v_path}${lang}"-lemm.raw" \
                     -output ${pretrain_w2v_path}${lang}"-lemm.w2v" \
                     -binary 1 -threads 4
            echo
            form_embed_dim=32
            ;;
    esac
    word2vec -size ${form_embed_dim} \
             -type 3 -hs 0 -min-count 2 -window 7 -sample 0.1 -negative 7 -iter 20 \
             -train ${pretrain_w2v_path}${lang}"-form.raw" \
             -output ${pretrain_w2v_path}${lang}"-form.w2v" \
             -binary 1 -threads 4
    echo
done
