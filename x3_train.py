def ready(lang, suffix):
    import src_conllu as conllu
    import src_ud2 as ud2
    from src_setup import Setup
    setup = Setup.load("./lab/setup/{}{}.npy".format(lang, suffix))
    dev = list(conllu.load(ud2.path(lang, 'dev')))

    def train(infix="", epochs=10, **kwargs):
        model = setup.model(**kwargs)
        for epoch in range(epochs):
            setup.train(model, verbose=2)
            conllu.save((setup.parse(model, sent) for sent in dev),
                        "./lab/result/{}{}-e{:0>2d}.conllu".format(lang, infix, epoch))

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

    hidden_init = 'he_uniform'
    if 0 == float(argv[2]):
        embed_init = 'he_uniform'
        output_init = 'he_uniform'
        infix = "-embed_init_he_uniform"
    else:
        embed_init = 'uniform'
        output_init = 'orthogonal'
        infix = "-output_init_orthogonal"

    # dense_init = argv[2]
    # infix = "-bn_{}"

    print(lang, suffix, infix)
    train = ready(lang, suffix)
    train(infix, 16,
          # hidden_layers=2,
          # hidden_units=256,

          # hidden_dropout=0,
          # hidden_bn=True,

          embed_init=embed_init,
          hidden_init=hidden_init,
          output_init=output_init,

          # embed_const='unitnorm',
          # hidden_const=None,
          # output_const=None,
    )
