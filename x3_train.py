def ready(lang, suffix):
    from darc import conllu, ud2
    from darc.setup import Setup
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

    norm = argv[2]

    bias = float(argv[3])

    infix = "-embed_const_{}-bias_{}".format(norm, bias)
    print(lang, suffix, infix)

    train = ready(lang, suffix)
    train(infix, 16,
          embed_const=norm,
          # hidden_layers=2,
          # hidden_units=256,
          hidden_bias=bias,
          # hidden_init='orthogonal',
          # hidden_const=None,
          # hidden_dropout=0.25,
          # activation='relu',
          # output_init='orthogonal',
          # output_const=None,
    ):
