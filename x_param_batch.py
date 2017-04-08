from setup import Setup
from conllu import load, save
# from keras import constraints as const

lang = "fa"
setup = Setup.load("./setups/fa-nonp.npy")

# lang = "grc_proiel"
# setup = Setup.load("./setups/grc_proiel-nonp.npy")

# lang = "zh"
# setup = Setup.load("./setups/zh-proj.npy")

model = setup.model()

dev = list(load("./golds/{}-ud-dev.conllu".format(lang)))

for batch_size in 6, 18, 32, 64:
    for epoch in range(10):
        setup.train(model, verbose=2, batch_size=batch_size)
        save([setup.parse(model, sent) for sent in dev],
             "./results/{}-batch_{}-e{:0>2d}.conllu"
             .format(lang, batch_size, epoch))
