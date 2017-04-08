from ud2 import lang_code
from conllu import load, save
from setup import Setup

ud_path = "/data/ud-treebanks-conll2017/"

for lang, code in lang_code:
    print("running on", lang, code, "...")
    dev_path = ud_path + "UD_{}/{}-ud-dev.conllu".format(lang, code)
    train_path = ud_path + "UD_{}/{}-ud-train.conllu".format(lang, code)
    embedding_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
                     "ud-2.0-baselinemodel-train-embeddings/" \
                     "{}.skip.forms.50.vectors".format(code)
    dev = list(load(dev_path))
    # save(dev, "./setups/{}-ud-dev.conllu".format(code))
    setup = Setup.make(train_path, embedding_path, proj=False)
    model = setup.model()
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        save(dev, "./results/{}_e{}.conllu"
             .format(code, epoch))
    del model
    del setup
    del dev
