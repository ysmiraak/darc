from setup import Setup
from conllu import load, save

lang, suffix = 'la_proiel', 'nonp'
# lang, suffix = 'fa', 'nonp'
# lang, suffix = 'grc_proiel', 'nonp'
# lang, suffix = 'zh', 'proj'

setup = Setup.load("./setups/{}-{}.npy".format(lang, suffix))
dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))


model = setup.model()
for epoch in range(10):
    setup.train(model, verbose=2)
    save((setup.parse(model, sent) for sent in dev),
         "./results/{}-e{:0>2d}.conllu"
         .format(lang, epoch))
