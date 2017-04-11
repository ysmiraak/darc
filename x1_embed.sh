w2v() {
    path=$1
    lang=$2
    attr=$3
    iter=$3
    negative=$4
    sample=$5
    $path"word2vec" \
         -type 3 \
         -hs 0 \
         -size 32 \
         -min-count 2 \
         -window 7 \
         -sample 1e-$sample \
         -negative $negative \
         -iter $iter \
         -threads 4 \
         -train $path$lang"."$attr".raw" \
         -output $path$lang"-s"$sample"n"$negative"i"$iter"."$attr \
         -binary 1
}

path="./embed/"
iter=16
negative=8
sample=1


lang="la_proiel"

w2v $path $lang "form" $iter $negative $sample
w2v $path $lang "lemm" $iter $negative $sample
