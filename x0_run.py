from setup import Setup
from conllu import load, write

ud_path = "/data/ud-treebanks-conll2017/"

# persian
lang = "fa"
proj = False
dev_path = ud_path + "UD_Persian/fa-ud-dev.conllu"
train_path = ud_path + "UD_Persian/fa-ud-train.conllu"

# # ancient greek proiel
# lang = "grc_proiel"
# proj = False
# dev_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"
# train_path = ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu"

# # chinese
# lang = "zh"
# proj = True
# dev_path = ud_path + "UD_Chinese/zh-ud-dev.conllu"
# train_path = ud_path + "UD_Chinese/zh-ud-train.conllu"

embedding_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
                 "ud-2.0-baselinemodel-train-embeddings/" \
                 "{}.skip.forms.50.vectors".format(lang)

setup = Setup.build(train_path, embedding_path, proj)

model = setup.model(
    # form_emb_reg=None,
    # form_emb_const='unit_norm',
    # upos_emb_dim=10,
    # upos_emb_reg=None,
    # upos_emb_const='unit_norm',
    # inputs_dropout=0.0,
    # hidden_units=200,
    # hidden_reg=None,
    # hidden_const=None,
    # hidden_dropout=0.0,
    # output_reg=None,
    # output_const=None,
    # optimizer='adamax'
)

# dev = list(load(dev_path))
# write(dev, "./golds/{}-ud-dev.conllu".format(lang))
dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))

for epoch in range(10):
    setup.train(model, verbose=2)
    write([setup.parse(model, sent) for sent in dev],
          "./results/{}-e{:0>2d}.conllu"
          .format(lang, epoch))
