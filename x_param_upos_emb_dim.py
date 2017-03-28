from nn_mlp import Setup
from conllu import load, write, validate

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

optimizer = 'adamax'
hidden_units = 200

for upos_emb_dim in 5, 10, 15, 20:
    model = setup.model(
        upos_emb_dim=upos_emb_dim,
        hidden_units=hidden_units,
        optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}d-upos_e{}.conllu"
              .format(upos_emb_dim, epoch))
