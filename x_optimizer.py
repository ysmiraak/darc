from nn_mlp import Setup
from conllu import load, write, validate

ud_path = "/data/ud-treebanks-conll2017/"

setup = Setup.load("./setups/grc_proiel-labeled.npy")

dev = list(load(ud_path + "/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"))

optimizers = 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'

for optimizer in optimizers:
    model = setup.model(optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}_e{}.conllu"
              .format(optimizer, epoch))
