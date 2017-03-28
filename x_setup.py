from nn_mlp import Setup
from conllu import load, write

ud_path = "/data/ud-treebanks-conll2017/"
wv_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
          "ud-2.0-baselinemodel-train-embeddings/"

write(load(ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"),
      "./setups/grc_proiel-ud-dev.conllu")

setup = Setup.build(
    ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu",
    wv_path + "grc_proiel.skip.forms.50.vectors",
    labeled=True)

setup.save("./setups/grc_proiel-labeled.npy")

# del setup

# setup = Setup.build(
#     ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu",
#     wv_path + "grc_proiel.skip.forms.50.vectors",
#     labeled=False)

# setup.save("./setups/grc_proiel-unlabeled.npy")
