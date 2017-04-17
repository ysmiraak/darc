import src_ud2 as ud2
import src_conllu as conllu
from src_setup import Setup
import json

silver_train_path = "./lab/silver_train/"
pretrain_w2v_path = "./lab/pretrain_w2v/"
system_model_path = "./lab/system_model/"


lang = "kk"

def make_setup(lang):
    silver = "{}{}.conllu".format(silver_train_path, lang)
    form_w2v = "{}{}-form.w2v".format(pretrain_w2v_path, lang)
    lemm_w2v = "{}{}-lemm.w2v".format(pretrain_w2v_path, lang) \
               if lang not in ud2.no_lemma else None
    return Setup.make(silver, form_w2v, lemm_w2v, proj=False)


setup = make_setup(lang)
model = setup.model()
setup.train(model, epochs=8)

dev = ud2.path(lang, 'train')

conllu.save((setup.parse(model, sent) for sent in conllu.load(dev)),
            "./lab/{}-system.conllu".format(lang))

setup.save("{}{}.npy".format(system_model_path, lang), model, with_data=False)

setup, model = Setup.load("{}{}.npy".format(system_model_path, lang), with_model=True)

conllu.save((setup.parse(model, sent) for sent in conllu.load(dev)),
            "./lab/{}-system2.conllu".format(lang))
