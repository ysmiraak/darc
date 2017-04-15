def setup(lang, proj=False, embed_path="./0000/embed/", setup_path="./0000/setup/"):
    from ud2 import path
    from setup import Setup
    from conllu import load, save
    # save(load(path(lang, 'dev')), "./0000/gold/{}-ud-dev.conllu".format(lang))
    Setup.make(path(lang, 'train'), proj=proj, binary=True,
               form_w2v="{}{}.form".format(embed_path, lang),
               lemm_w2v="{}{}.lemm".format(embed_path, lang)) \
         .save("{}{}-{}.npy".format(setup_path, lang, 'proj' if proj else 'nonp'))


if '__main__' == __name__:
    from sys import argv
    lang = argv[1]
    try:
        proj = bool(argv[2])
    except IndexError:
        proj = False
    print(lang, proj)
    setup(lang, proj)
