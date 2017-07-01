import src_conllu as conllu
import src_ud2 as ud2

from lab_setup6 import Setup
trial = 6


def train_parse(lang):
    setup = Setup.make(ud2.path(lang))
    model = setup.model()
    sents = list(conllu.load(ud2.path(lang, ds="dev")))
    for epoch in range(25):
        setup.train(model, verbose=2)
        if 4 <= epoch:
            conllu.save((setup.parse(model, sent) for sent in sents)
                        , "./lab/{}_{}_{}.conllu".format(lang, trial, epoch))


if '__main__' == __name__:
    from lab import langs
    for lang in langs:
        train_parse(lang)
