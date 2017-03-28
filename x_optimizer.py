from nn_mlp import Setup
from conllu import load, write, validate

setup = Setup.load("./setups/grc_proiel-labeled.npy")

dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

# optimizers = 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'

optimizers = 'adadelta', 'adam', 'adamax'

for optimizer in optimizers:
    model = setup.model(optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/{}_e{}.conllu"
              .format(optimizer, epoch))
