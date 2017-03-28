from nn_mlp import Setup
from conllu import load, write, validate

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

optimizer = 'adamax'

for hidden_units in 50, 100, 150, 200, 250, 300:
    model = setup.model(hidden_units=hidden_units, optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}units_e{}.conllu"
              .format(hidden_units, epoch))
