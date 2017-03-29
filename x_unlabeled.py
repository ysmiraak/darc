from nn_mlp import Setup
from conllu import load, write, validate

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-unlabeled.npy")

hidden_units = 200

model = setup.model(
    # form_emb_reg=None,
    # form_emb_const='unit_norm',
    # upos_emb_dim=10,
    # upos_emb_reg=None,
    # upos_emb_const='unit_norm',
    # inputs_dropout=0.0,
    hidden_units=hidden_units,
    # hidden_reg=None,
    # hidden_const=None,
    # hidden_dropout=0.0,
    # output_reg=None,
    # output_const=None,
    # optimizer='adamax'
)

for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "./results/{}units_e{}.conllu"
          .format(hidden_units, epoch))
