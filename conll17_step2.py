from src_setup import Setup
import src_ud2 as ud2
import numpy as np

silver_train_path = "./lab/silver_train/"
pretrain_w2v_path = "./lab/pretrain_w2v/"
system_model_path = "./lab/system_model/"


def make_setup(lang, proj):
    """-> Setup"""
    silver = "{}{}.conllu".format(silver_train_path, lang)
    form_w2v = "{}{}-form.w2v".format(pretrain_w2v_path, lang)
    lemm_w2v = "{}{}-lemm.w2v".format(pretrain_w2v_path, lang) \
               if lang not in ud2.no_lemma else None
    return Setup.make(silver, form_w2v, lemm_w2v, proj=proj)


def train_save(setup, suffix):
    """save the setup and model weights from epoch 4 to 16"""
    setup.save("{}{}{}.npy".format(system_model_path, lang, suffix), with_data=False)
    model = setup.model()
    for epoch in range(16):
        setup.train(model, verbose=2)
        if 4 <= epoch:
            np.save("{}{}{}-e{:0>2d}.npy".format(system_model_path, lang, suffix, epoch),
                    {'model': model.to_json(), 'weights': model.get_weights()})


if '__main__' == __name__:
    lang = argv[1]
    train_save(make_setup(lang, False), "-nonp")
    train_save(make_setup(lang, True), "-proj")
