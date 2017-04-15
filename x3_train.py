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


if '__main__' == __name__:
    from sys import argv

    if 'la' == argv[1]:
        lang, suffix = 'la_proiel', '-nonp'
    else:
        lang, suffix = 'he', '-proj'
    # lang, suffix = 'fa', 'nonp'
    # lang, suffix = 'grc_proiel', 'nonp'
    # lang, suffix = 'zh', 'proj'

    hidden = int(argv[2])
    dropout = float(argv[3])

    infix = "-hidden_{}-dropout_{}-relu".format(hidden, dropout)
    print(lang, infix)

    train = ready(lang, suffix)
    train(infix, 16,
          hidden_units=hidden,
          dense_dropout=dropout,
          activation='relu')
