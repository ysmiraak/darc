from ud2 import ud_path, treebanks
from setup import Setup
from conllu import load, save


def setup(lang, suffix="", proj=False):
    dev = ud_path + "UD_{}/{}-ud-dev.conllu".format(treebanks[lang], lang)
    save(load(dev), "./golds/{}-ud-dev.conllu".format(lang))
    train = ud_path + "UD_{}/{}-ud-train.conllu".format(treebanks[lang], lang)
    form_w2v = "./embed/{}-form{}.w2v".format(lang, suffix)
    lemm_w2v = "./embed/{}-lemm{}.w2v".format(lang, suffix)
    Setup.make(train, form_w2v, lemm_w2v, proj) \
         .save("./setups/{}-{}{}.npy"
               .format(lang, 'proj' if proj else 'nonp', suffix))


if '__main__' == __name__:
    from sys import argv
    try:
        argv[3] = bool(argv[3])
    except IndexError:
        pass
    setup(*argv[1:])
