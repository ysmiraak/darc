from nn_mlp import Setup
from conllu import load, write, validate
from keras import constraints as const

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

axis = 0

model = setup.model(
    form_emb_const=const.unit_norm(axis),
    upos_emb_const=const.unit_norm(axis))

for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "./results/unit_norm_e{}.conllu"
          .format(epoch))
