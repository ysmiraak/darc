from setup import Setup
from conllu import load, save

lang, suffix = 'la_proiel', 'nonp'
# lang, suffix = 'fa', 'nonp'
# lang, suffix = 'grc_proiel', 'nonp'
# lang, suffix = 'zh', 'proj'

setup = Setup.load("./setups/{}-{}.npy".format(lang, suffix))
dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))


emb_dropouts = 0.1, 0.15, 0.2, 0.25, 0.3, 0.35
emb_dropout = emb_dropouts[0]

model = setup.model(emb_dropout=emb_dropout)
for epoch in range(10):
    setup.train(model, verbose=2)
    save((setup.parse(model, sent) for sent in dev),
         "./results/{}-emb_dropout={}-e{:0>2d}.conllu"
         .format(lang, emb_dropout, epoch))
