from ud2 import path
from setup import Setup
from conllu import load, save


def ready(lang, suffix):
    setup = Setup.load("./setups/{}{}.npy".format(lang, suffix))
    dev = list(load(path(lang, 'dev')))

    def train(infix="", epochs=10, **kwargs):
        model = setup.model(**kwargs)
        for epoch in range(epochs):
            setup.train(model, verbose=2)
            save((setup.parse(model, sent) for sent in dev),
                 "./results/{}{}-e{:0>2d}.conllu".format(lang, infix, epoch))

    return train


lang, suffix = 'la_proiel', '-nonp'
# lang, suffix = 'fa', 'nonp'
# lang, suffix = 'grc_proiel', 'nonp'
# lang, suffix = 'zh', 'proj'

w = 4, 5, 6, 7, 8
infix = "-w{}s1n8i16".format(w[0])
train = ready(lang, suffix + infix)
train(infix, 16)
