# l1-normalized summed embedding, dumb root, mask zero, 32 dim

import src_conllu as conllu
from src_conllu import Sent
from src_transition import Config, Oracle
import numpy as np
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Reshape, Lambda
from keras.initializers import uniform


class Setup(object):
    """for dependency parsing with upostag, feats, and deprel"""
    __slots__ = 'idx2tran', 'upos2idx', 'drel2idx', 'feat2idx', 'x', 'y'

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    @staticmethod
    def cons(sents):
        """[Sent] -> Setup"""
        # upos2idx feat2idx drel2idx idx2tran
        upos2idx = {Sent.dumb: 0, Sent.root: 1, 'X': 2}
        feat2idx = {Sent.dumb: 0, Sent.root: 1}
        drel2idx = {Sent.dumb: 0}
        idx2tran = [('shift', None), ('swap', None)]
        if not hasattr(sents, '__len__'):
            sents = list(sents)
        for sent in sents:
            it = zip(sent.upostag, sent.feats, sent.deprel)
            next(it)
            for upos, feats, drel in it:
                if upos not in upos2idx:
                    upos2idx[upos] = len(upos2idx)
                for feat in feats.split("|"):
                    if feat not in feat2idx:
                        feat2idx[feat] = len(feat2idx)
                if drel not in drel2idx:
                    drel2idx[drel] = len(drel2idx)
                    idx2tran.append(('left', drel))
                    idx2tran.append(('right', drel))
        feat2idx[Sent.dumb] = len(feat2idx)  # free idx 0 for mask
        # x y
        self = Setup(upos2idx=upos2idx, drel2idx=drel2idx, feat2idx=feat2idx, idx2tran=idx2tran)
        tran_idx = np.eye(len(idx2tran), dtype=np.float32)
        tran2idx = {tran: tran_idx[idx] for idx, tran in enumerate(idx2tran)}
        data = [],     [],     [],     []
        name = "upos", "drel", "feat"
        upos_append, drel_append, feat_append, tran_append, = (d.append for d in data)
        for sent in sents:
            oracle = Oracle.cons(sent)
            config = Config.cons(sent)
            while not config.is_terminal():
                tran = oracle.predict(config)
                if not config.doable(tran[0]):
                    # this happends on a non-proj sent with proj setting
                    break
                feature = self.feature(config, named=False)
                upos_append(feature[0])
                drel_append(feature[1])
                feat_append(feature[2])
                tran_append(tran2idx[tran])
                getattr(config, tran[0])(tran[1])
        self.x = {n: np.concatenate(d) for n, d in zip(name, data)}
        self.y = np.array(data[-1], np.float32)
        return self

    @staticmethod
    def make(train_conllu):
        """-> Setup; from files"""
        return Setup.cons(sents=conllu.load(train_conllu))

    def model(self, feat_emb_dim=32):
        """-> keras.models.Model"""
        embed_init = uniform(minval=-0.5, maxval=0.5)
        # all possible inputs
        upos = Input(name="upos", shape=self.x["upos"].shape[1:], dtype=np.uint8)
        drel = Input(name="drel", shape=self.x["drel"].shape[1:], dtype=np.uint8)
        feat = Input(name="feat", shape=self.x["feat"].shape[1:], dtype=np.uint8)
        # cons layers
        i = [upos, drel, feat]
        upos = Flatten(name="upos_flat")(
            Embedding(
                input_dim=len(self.upos2idx),
                output_dim=12,
                embeddings_initializer=embed_init,
                name="upos_embed")(upos))
        drel = Flatten(name="drel_flat")(
            Embedding(
                input_dim=len(self.drel2idx),
                output_dim=16,
                embeddings_initializer=embed_init,
                name="drel_embed")(drel))
        feat = Embedding(
                    input_dim=1 + len(self.feat2idx),
                    output_dim=feat_emb_dim,
                    embeddings_initializer=embed_init,
                    mask_zero=True,
                    name="feat_embed")(feat)
        def normalized_sum(x):
            x = K.reshape(x, (-1, 18, len(self.feat2idx), feat_emb_dim))
            x = K.sum(x, -2)
            norm = K.maximum(K.sum(K.abs(x), -1), 1e-12)
            norm = K.reshape(norm, (-1, 18, 1))
            x /= norm
            return x
        feat = Lambda(normalized_sum, name="feat_aggr")(feat)
        feat = Flatten(name="feat_flat")(feat)
        o = [upos, drel, feat]
        o = Concatenate(name="concat")(o)
        for hid in range(2):
            o = Dense(
                units=256,
                activation='relu',
                kernel_initializer='he_uniform',
                name="hidden{}".format(1 + hid))(o)
        o = Dense(
            units=len(self.idx2tran),
            activation='softmax',
            kernel_initializer='he_uniform',
            name="output")(o)
        m = Model(i, o, name="darc")
        m.compile('adamax', 'categorical_crossentropy')
        return m

    def train(self, model, *args, **kwargs):
        """mutates model by calling keras.models.Model.fit"""
        model.fit(self.x, self.y, *args, **kwargs)

    def parse(self, model, sent):
        """-> Sent"""
        config = Config.cons(sent)
        while not config.is_terminal():
            if 2 > len(config.stack):
                config.shift()
                continue
            prob = model.predict(self.feature(config), 1).ravel()
            for r in prob.argsort()[::-1]:
                act, arg = self.idx2tran[r]
                if config.doable(act):
                    getattr(config, act)(arg)
                    break
            else:
                print("WARNING!!!! FAILED TO PARSE:", " ".join(sent.form))
                break
        return config.finish()

    def feature(self, config, named=True):
        """-> [numpy.ndarray] :as form, upos, drel, feat

        assert form.shape == upos.shape == (18, )

        assert drel.shape == (16, )

        assert feat.shape == (18 * len(self.feat2idx), )

        """
        # 18 features (Chen & Manning 2014)
        #  0: s0       1: s1       2: s2
        #  3: s0l1     4: s1l1     5: s0r1     6: s1r1
        #  7: s0l0     8: s1l0     9: s0r0    10: s1r0
        # 11: s0l0l1  12: s1l0l1  13: s0r0r1  14: s1r0r1
        # 15: i0      16: i1      17: i2
        i, s, g = config.input, config.stack, config.graph
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        # node 0 in each sent is dumb
        len_s = len(s)
        if 1 <= len_s:
            x[0] = s[-1]  # s0
            y = g[x[0]]
            if y:
                if 2 <= len(y):
                    x[3] = y[1]  # s0l1
                    x[5] = y[-2]  # s0r1
                x[7] = y[0]  # s0l0
                x[9] = y[-1]  # s0r0
                y = g[x[7]]
                if y:
                    x[11] = y[0]  # s0l0l1
                y = g[x[9]]
                if y:
                    x[13] = y[-1]  # s0r0r1
            if 2 <= len_s:
                x[1] = s[-2]  # s1
                y = g[x[1]]
                if y:
                    if 2 <= len(y):
                        x[4] = y[1]  # s1l1
                        x[6] = y[-2]  # s1r1
                    x[8] = y[0]  # s1l0
                    x[10] = y[-1]  # s1r0
                    y = g[x[8]]
                    if y:
                        x[12] = y[0]  # s1l0l1
                    y = g[x[10]]
                    if y:
                        x[14] = y[-1]  # s1r0r1
                if 3 <= len_s:
                    x[2] = s[-3]  # s2
        len_i = len(i)
        if 1 <= len_i:
            x[15] = i[-1]  # i0
            if 2 <= len_i:
                x[16] = i[-2]  # i1
                if 3 <= len_i:
                    x[17] = i[-3]  # i2
        # upos
        upos2idx = self.upos2idx.get
        upos_unk = upos2idx('X')  # upos is never _
        upos = config.sent.upostag
        upos = np.fromiter((upos2idx(upos[i], upos_unk) for i in x), np.uint8)
        # drel
        drel2idx = self.drel2idx
        drel = config.deprel
        drel = np.fromiter((drel2idx[drel[i]] for i in x[2:-4]), np.uint8)
        # feats
        feats = config.sent.feats
        feats = [feats[i] for i in x]
        # special treatments for root
        if 3 >= len_s:
            r = len_s - 1
            root = Sent.root
            upos[r] = upos2idx(root)
            feats[r] = root
        # set-valued feat (Alberti et al. 2015)
        feat2idx = self.feat2idx
        feat = np.zeros((len(feats), len(feat2idx)), np.uint8)
        for ftv, fts in zip(feat, feats):
            for ft in fts.split("|"):
                try:
                    idx = feat2idx[ft]
                except KeyError:
                    pass
                else:
                    ftv[idx - 1] = idx
        upos.shape = drel.shape = feat.shape = 1, -1
        if named:
            return {'upos': upos, 'drel': drel, 'feat': feat}
        else:
            return [upos, drel, feat]

    def save(self, file, model=None, with_data=True):
        """as npy file"""
        bean = {attr: getattr(self, attr) for attr in
                (Setup.__slots__ if with_data else Setup.__slots__[:-4])}
        if model is not None:
            bean['model'] = model.to_json()
            bean['weights'] = model.get_weights()
        np.save(file, bean)

    @staticmethod
    def load(file, with_model=False):
        """str, False -> Setup; str, True -> Setup, keras.models.Model"""
        bean = np.load(file).item()
        if with_model:
            model = model_from_json(bean['model'])
            model.set_weights(bean['weights'])
            del bean['weights']
            del bean['model']
            return Setup(**bean), model
        else:
            return  Setup(**bean)


if '__main__' == __name__:

    trial = 8

    from lab import langs
    import src_ud2 as ud2

    def train_parse(lang):
        setup = Setup.make(ud2.path(lang))
        model = setup.model()
        sents = list(conllu.load(ud2.path(lang, ds="dev")))
        for epoch in range(25):
            setup.train(model, verbose=2)
            conllu.save((setup.parse(model, sent) for sent in sents)
                        , "./lab/{}-t{}-e{:02d}.conllu".format(lang, trial, epoch))

    for lang in langs:
        train_parse(lang)
