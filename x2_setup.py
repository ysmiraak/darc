from ud2 import path
from setup import Setup
from conllu import load, save


def setup(lang, suffix="", proj=False, embed_path="./embed/", setup_path="./setups/"):
    # save(load(path(lang, 'dev')), "./golds/{}-ud-dev.conllu".format(lang))
    Setup.make(path(lang, 'train'), proj=proj, binary=True,
               form_w2v="{}{}{}.form".format(embed_path, lang, suffix),
               lemm_w2v="{}{}{}.lemm".format(embed_path, lang, suffix)) \
         .save("{}{}-{}{}.npy".format(setup_path, lang, 'proj' if proj else 'nonp', suffix))


if '__main__' == __name__:
    # from sys import argv
    # try:
    #     argv[3] = bool(argv[3])
    # except IndexError:
    #     pass
    # setup(*argv[1:])
    lang = 'la_proiel'
    for w in 4, 5, 6, 7, 8:
        suffix = "-w{}s1n8i16".format(w)
        print(lang, suffix)
        setup(lang, suffix)
