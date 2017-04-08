from setup import Setup
from conllu import load, write


lang = "fa"
setup = Setup.load("./setups/fa-nonp.npy")


dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))

for upos_emb_dim in 8, 9, 10, 11, 12:
    for drel_emb_dim in 14, 15, 16, 17, 18:
        model = setup.model(
            upos_emb_dim=upos_emb_dim,
            drel_emb_dim=drel_emb_dim,
        )
        for epoch in range(10):
            setup.train(model, verbose=2)
            write([setup.parse(model, sent) for sent in dev],
                  "./results/{}-{:0>2d}_upos-{:0>2d}_drel-e{}.conllu"
                  .format(lang, upos_emb_dim, drel_emb_dim, epoch))
        del model
