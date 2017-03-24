from nn_mlp import Setup
from conllu import load, write, validate

ud_path = "/data/ud-treebanks-conll2017/"
wv_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
          "ud-2.0-baselinemodel-train-embeddings/"

# setup = Setup.build(
#     ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu",
#     wv_path + "grc_proiel.skip.forms.50.vectors",
#     labeled=True)
# setup.save("./setups/grc_proiel-labeled.npy")
# del setup
# setup = Setup.build(
#     ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu",
#     wv_path + "grc_proiel.skip.forms.50.vectors",
#     labeled=False)
# setup.save("./setups/grc_proiel-unlabeled.npy")

# setup = Setup.load("./setups/grc_proiel-unlabeled.npy")

setup = Setup.load("./setups/grc_proiel-labeled.npy")

dev = list(load(ud_path + "/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"))

# # search for optimizer
# for optimizer in ('sgd', 'rmsprop', 'adagrad', 'adadelta',
#                   'adam', 'adamax', 'nadam'):
#     model = setup.model(optimizer=optimizer)
#     for epoch in range(10):
#         setup.train(model, verbose=2)
#         for sent in dev:
#             setup.parse(model, sent)
#         validate(dev)
#         write(dev, "./results/{}_{}.conllu".format(optimizer, epoch + 1))

# # search for hidden units
# optimizer = 'adamax'
# for hidden_units in 50, 100, 150, 200, 250, 300:
#     model = setup.model(hidden_units=hidden_units, optimizer=optimizer)
#     for epoch in range(10):
#         setup.train(model, verbose=2)
#         for sent in dev:
#             setup.parse(model, sent)
#         validate(dev)
#         write(dev, "./results/{}_{}units_{}.conllu"
#               .format(optimizer, hidden_units, 1 + epoch))

# search for upos_emb_dim
optimizer = 'adamax'
hidden_units = 200
for upos_emb_dim in 5, 10, 15, 20:
    model = setup.model(hidden_units=hidden_units, optimizer=optimizer)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, "./results/upos-dim_{}_epoch_{}.conllu"
              .format(upos_emb_dim, epoch))
