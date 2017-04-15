def ready(lang, suffix):
    from ud2 import path
    from setup import Setup
    from conllu import load, save
    setup = Setup.load("./0000/setup/{}{}.npy".format(lang, suffix))
    dev = list(load(path(lang, 'dev')))

    def train(infix="", epochs=10, **kwargs):
        model = setup.model(**kwargs)
        for epoch in range(epochs):
            setup.train(model, verbose=2)
            save((setup.parse(model, sent) for sent in dev),
                 "./0000/result/{}{}-e{:0>2d}.conllu".format(lang, infix, epoch))

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

    infix = "-embed_const_{}".format(norm)
    print(lang, suffix, infix)

    train = ready(lang, suffix)
    train(infix, 16,
          embed_const=norm,
          # hidden_layers=2,
          # hidden_units=256,
          # hidden_bias='zeros',
          # hidden_init='orthogonal',
          # hidden_const=None,
          # hidden_dropout=0.25,
          # activation='relu',
          # output_init='orthogonal',
          # output_const=None,
    ):
