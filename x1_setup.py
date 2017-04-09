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

save(load(dev_path), "./golds/{}-ud-dev.conllu".format(lang))
Setup.make(train_path, embedding_path, proj) \
     .save("./setups/{}-{}.npy" .format(lang, 'proj' if proj else 'nonp'))
