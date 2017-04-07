from setup import Setup
from conllu import load, write
# from keras import constraints as const


inits = 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_uniform'
init = inits[0]


lang = "fa"
setup = Setup.load("./setups/fa-nonp.npy")

# lang = "grc_proiel"
# setup = Setup.load("./setups/grc_proiel-nonp.npy")

# lang = "zh"
# setup = Setup.load("./setups/zh-proj.npy")


model = setup.model(
    # upos_emb_dim=10,
    # drel_emb_dim=15,
    # emb_init=emb_init,
    # emb_const=unit_norm(),
    # emb_dropout=0.0,
    # hidden_units=200,
    hidden_init=init,
    # hidden_const=None,
    # hidden_dropout=0.0,
    # output_init='glorot_uniform',
    # output_const=None,
    # activation='tanh',
    # optimizer='adamax',
)


dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))

for epoch in range(10):
    setup.train(model, verbose=2)
    write([setup.parse(model, sent) for sent in dev],
          "./results/{}-hidden_{}-e{:0>2d}.conllu"
          .format(lang, init, epoch))
