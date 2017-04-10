import numpy as np
from keras.models import model_from_json


class Model(object):
    __slots__ = 'setup', 'model'

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    def parse(self, sent):
        """-> Sent"""
        return self.setup.parse(self.model, sent)

    @staticmethod
    def cons():
        pass

    @staticmethod
    def make():
        pass

    def save(self, file, with_data=False):
        """as npy file"""
        np.save(file, {
            'setup': self.setup.bean(with_data),
            'model': self.model.to_json(),
            'weights': self.model.get_weights()
        })

    @staticmethod
    def load(file):
        """-> Model"""
        bean = np.load(file).item()
        model = model_from_json(bean['model'])
        model.set_weights(bean['weights'])
        return Model(setup=bean['setup'], model=model)
