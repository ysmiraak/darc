from nn_mlp import Setup
from conllu import load, write, validate

ud_path = "/data/ud-treebanks-conll2017/UD_Ancient_Greek-PROIEL/"
wv_path = ("/data/udpipe-ud-2.0-conll17-170315-supplementary-data/"
           "ud-2.0-baselinemodel-train-embeddings/")

setup = Setup.build(ud_path + "grc_proiel-ud-train.conllu",
                    wv_path + "grc_proiel.skip.forms.50.vectors")

dev = list(load(ud_path + "grc_proiel-ud-dev.conllu"))

for optimizer in ('sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'):
    print("try optimizer", optimizer, "...")
    model = setup.model(optimizer=optimizer)
    for epoch in range(10):
        print("train in epoch", epoch + 1, "...")
        setup.train(model)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}_{}.conllu".format(optimizer, epoch + 1))
        print("\n")
    print("\n\n\n")
