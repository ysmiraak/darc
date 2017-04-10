from conllu import load, save
from setup import Setup
from setup_no_lemma import SetupNoLemma
import numpy as np
from keras.models import model_from_json


class Model(object):
    """parser model"""
    __slots__ = 'setup', 'model'

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def parse_conllu(self, input_file, out_file):
        """does the darc"""
        save((self.setup.parse(sent, self.model) for sent in load(input_file)),
             out_file)

    def parse(self, sent):
        """-> Sent"""
        return self.setup.parse(self.model, sent)

    def train(self, verbose=1, epochs=12):
        """mutates model by calling keras.models.Model.fit"""
        self.setup.train(model, verbose=verbose, epochs=epochs)

    @staticmethod
    def cons(setup):
        """-> Model"""
        self.model = setup.model()

    @staticmethod
    def make(train_conllu, form_w2v, lemm_w2v=None, binary=True, proj=False):
        """-> Model; from files"""
        if lemm_w2v:
            setup = Setup.make(train_conllu, form_w2v, lemm_w2v, binary)
        else:
            setup = SetupNoLemma.make(train_conllu, form_w2v, binary)
        return Model.cons(setup)

    def save(self, file, with_data=False):
        """as npy file"""
        np.save(file,
                {'setup': self.setup.bean(with_data),
                 'model': self.model.to_json(),
                 'weights': self.model.get_weights()})

    @staticmethod
    def load(file):
        """-> Model"""
        bean = np.load(file).item()
        model = model_from_json(bean['model'])
        model.set_weights(bean['weights'])
        setup = Setup if 'lemm2idx' in bean['setup'] else SetupNoLemma
        return Model(setup=setup(bean['setup']), model=model)
