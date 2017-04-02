from nn_mlp import Setup
from conllu import load, write

ud_path = "/data/ud-treebanks-conll2017/"


# ancient greek proiel
lang = "grc_proiel"
projective = False
labeled = True
dev_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"
train_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu"


# # chinese
# lang = "zh"
# projective = True
# labeled = True
# dev_path = ud_path + "UD_Chinese/zh-ud-dev.conllu"
# train_path = ud_path + "UD_Chinese/zh-ud-train.conllu"


write(load(dev_path), "./golds/{}-ud-dev.conllu".format(lang))

embedding_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
                 "ud-2.0-baselinemodel-train-embeddings/" \
                 "{}.skip.forms.50.vectors".format(lang)

setup = Setup.build(
    train_path,
    embedding_path,
    labeled=labeled,
    projective=projective
)

setup.save("./setups/{}_{}_{}.npy"
           .format(lang,
                   '+proj' if projective else '-proj',
                   '+label' if labeled else '-label'))
