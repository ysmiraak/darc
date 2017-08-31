# from thesis_atomic import Setup
# from thesis_binary import Setup
# from thesis_onehot import Setup
from thesis_summed import Setup

if '__main__' == __name__:

    trial = 0
    use_form = False
    use_lemma = False
    embed_const = None
    embed_dropout = 0

    train_path = "./thesis/train/"
    embed_path = "./thesis/embed/"
    parse_path = "./thesis/parse/"

    def train_parse(lang):
        setup = Setup.make(
            "{}{}-ud-train.conllu".format(train_path, lang)
            , form_w2v="{}{}-form{}.w2v".format(embed_path, lang, 32 if use_lemma else 64) if use_form else None
            , lemm_w2v="{}{}-lemm32.w2v".format(embed_path, lang) if use_lemma else None
            , binary=True
            , proj=False)
        model = setup.model(embed_const=embed_const, embed_dropout=embed_dropout)
        sents = list(conllu.load("{}{}-ud-dev.conllu".format(train_path, lang)))
        for epoch in range(25):
            setup.train(model, verbose=2)
            conllu.save((setup.parse(model, sent) for sent in sents)
                        , "{}{}-t{:02d}-e{:02d}.conllu".format(parse_path, lang, trial, epoch))

    for lang in 'ar bg eu fa fi_ftb grc he hr it la_proiel nl pl sv tr zh'.split():
        train_parse(lang)
