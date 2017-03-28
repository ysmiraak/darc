from nn_mlp import Setup
from conllu import load, write, validate
from keras import constraints as const

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

model = setup.model(
    # form_emb_reg=None,
    form_emb_const=const.max_norm(1.0),
    # upos_emb_dim=10,
    # upos_emb_reg=None,
    upos_emb_const=const.max_norm(1.0),
    # hidden_units=200,
    # hidden_reg=None,
    hidden_const=const.max_norm(1.0),
    # output_reg=None,
    # output_const=None,
    # optimizer='adamax'
)

for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "./results/const_e{}.conllu"
          .format(epoch))
