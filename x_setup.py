from nn_mlp import Setup
from conllu import load, write

ud_path = "/data/ud-treebanks-conll2017/"


# ancient greek proiel
lang = "grc_proiel"
dev_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"
train_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu"

# # chinese
# lang = "zh"
# dev_path = ud_path + "UD_Chinese/zh-ud-dev.conllu"
# train_path = ud_path + "UD_Chinese/zh-ud-train.conllu"


write(load(dev_path), "./setups/{}-ud-dev.conllu".format(lang))

embedding_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
                 "ud-2.0-baselinemodel-train-embeddings/" \
                 "{}.skip.forms.50.vectors".format(lang)

setup = Setup.build(
    train_path,
    embedding_path,
    labeled=True,
    projective=True
)

setup.save("./setups/{}-labeled.npy".format(lang))

del setup

setup = Setup.build(
    train_path,
    embedding_path,
    labeled=False,
    projective=True
)

setup.save("./setups/{}-unlabeled.npy".format(lang))
