from nn_mlp import Setup
from conllu import load, write, validate

ud_path = "/data/ud-treebanks-conll2017/"

setup = Setup.load("./setups/grc_proiel-labeled.npy")

dev = list(load(ud_path + "/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"))

optimizer = 'adamax'

for hidden_units in 50, 100, 150, 200, 250, 300:
    model = setup.model(hidden_units=hidden_units, optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}unit_e{}.conllu"
              .format(hidden_units, epoch))
