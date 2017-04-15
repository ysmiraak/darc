from ud2 import path
from setup import Setup
from conllu import load, save


def setup(lang, proj=False, embed_path="./embeddings/", setup_path="./new_setups/"):
    # save(load(path(lang, 'dev')), "./golds/{}-ud-dev.conllu".format(lang))
    Setup.make(path(lang, 'train'), proj=proj, binary=True,
               form_w2v="{}{}.form".format(embed_path, lang),
               lemm_w2v="{}{}.lemm".format(embed_path, lang)) \
         .save("{}{}-{}.npy".format(setup_path, lang, 'proj' if proj else 'nonp'))


if '__main__' == __name__:
    from sys import argv
    try:
        argv[2] = bool(argv[2])
    except IndexError:
        pass
    setup(*argv[1:])
