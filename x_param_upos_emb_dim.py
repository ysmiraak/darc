from nn_mlp import Setup
from conllu import load, write, validate

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

for upos_emb_dim in 5, 15, 17, 20:
    model = setup.model(
        setup,
        # form_emb_reg=None,
        # form_emb_const='unit_norm',
        upos_emb_dim=upos_emb_dim,
        # upos_emb_reg=None,
        # upos_emb_const='unit_norm',
        # hidden_units=200,
        # hidden_reg=None,
        # hidden_const=None,
        # output_reg=None,
        # output_const=None,
        # optimizer='adamax'
    )
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}d-upos_e{}.conllu"
              .format(upos_emb_dim, epoch))
    del model
