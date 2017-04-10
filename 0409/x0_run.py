from ud2 import ud_path, data_path, treebanks
from setup import Setup
from conllu import load, save

lang, proj = 'la_proiel', False
# lang, proj = 'fa', False
# lang, proj = 'grc_proiel', False
# lang, proj = 'zh', True

dev_path = ud_path + "UD_{}/{}-ud-dev.conllu".format(treebanks[lang], lang)
train_path = ud_path + "UD_{}/{}-ud-train.conllu".format(treebanks[lang], lang)
embedding_path = data_path + "{}.skip.forms.50.vectors".format(lang)

setup = Setup.make(train_path, embedding_path, proj)
dev = list(load(dev_path))
# save(dev, "./golds/{}-ud-dev.conllu".format(lang))


model = setup.model()
for epoch in range(10):
    setup.train(model, verbose=2)
    save((setup.parse(model, sent) for sent in dev),
         "./results/{}-e{:0>2d}.conllu"
         .format(lang, epoch))
