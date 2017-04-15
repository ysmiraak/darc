w2v() {
    path=$1
    lang=$2
    attr=$3
    word2vec \
        -type 3 \
        -hs 0 \
        -size 32 \
        -min-count 2 \
        -window 7 \
        -sample 0.1 \
        -negative 7 \
        -iter 20 \
        -threads 4 \
        -train $path$lang"."$attr".raw" \
        -output $path$lang"."$attr \
        -binary 1
}

path="./lab/embed/"

# lang="la_proiel"
# lang="he"
lang=$1

w2v $path $lang "form"
w2v $path $lang "lemm"
