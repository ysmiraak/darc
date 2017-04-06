from nn_mlp import Setup
from conllu import load, write
# from keras import constraints as const


dev = list(load("./golds/fa-ud-dev.conllu"))
setup = Setup.load("./setups/fa-nonp.npy")

# dev = list(load("./golds/grc_proiel-ud-dev.conllu"))
# setup = Setup.load("./setups/grc_proiel_-proj_+label.npy")

# dev = list(load("./golds/zh-ud-dev.conllu"))
# setup = Setup.load("./setups/zh_-proj_+label.npy")


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


for epoch in range(10):
    setup.train(model, verbose=2)
    write([setup.parse(model, sent) for sent in dev],
          "./results/e{:0>2d}.conllu"
          .format(epoch))
