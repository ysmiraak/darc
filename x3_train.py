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

    inits = "orthogonal", "he_normal", "he_uniform"
    init = inits[int(argv[2])]
    
    infix = "-init_{}".format(init)

    print(lang, suffix, infix)
    train = ready(lang, suffix)
    train(infix, 16,
          # hidden_layers=2,
          # hidden_units=256,

          init=init,

          # hidden_const=None,
          # output_const=None,
    )
