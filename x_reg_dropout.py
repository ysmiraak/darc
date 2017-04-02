from nn_mlp import Setup
from conllu import load, write

dev = list(load("./golds/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel_-proj_+label.npy")

hidden_units = 200

model = setup.model(
    # form_emb_reg=None,
    # form_emb_const='unit_norm',
    # upos_emb_dim=10,
    # upos_emb_reg=None,
    # upos_emb_const='unit_norm',
    # inputs_dropout=0.5,
    hidden_units=hidden_units,
    # hidden_reg=None,
    # hidden_const=None,
    hidden_dropout=0.5,
    # output_reg=None,
    # output_const=None,
    # optimizer='adamax'
)

for epoch in range(25):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    write(dev, "./results/dropout_{}_e{:0>2d}.conllu"
          .format(hidden_units, epoch))
