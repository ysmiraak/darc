def setup(lang, proj=False, embed_path="./lab/embed/", setup_path="./lab/setup/"):
    from darc import ud2
    from darc.setup import Setup
    # from darc import conllu
    # conllu.save(conllu.load(ud2.path(lang, 'dev')), "./lab/gold/{}-ud-dev.conllu".format(lang))
    Setup.make(ud2.path(lang, 'train'), proj=proj, binary=True,
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
