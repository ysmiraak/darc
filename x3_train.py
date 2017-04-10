from ud2 import path
from setup import Setup
from conllu import load, save

lang, suffix = 'la_proiel', '-nonp'
# lang, suffix = 'fa', 'nonp'
# lang, suffix = 'grc_proiel', 'nonp'
# lang, suffix = 'zh', 'proj'

setup = Setup.load("./setups/{}{}.npy".format(lang, suffix))
dev = list(load(path(lang, 'dev')))


def train(infix="", epochs=10, **kwargs):
    model = setup.model(**kwargs)
    for epoch in range(epochs):
        setup.train(model, verbose=2)
        save((setup.parse(model, sent) for sent in dev),
             "./results/{}{}-e{:0>2d}.conllu"
             .format(lang, infix, epoch))
