import src_conllu as conllu
from src_conllu import Sent
from src_transition import Config, Oracle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dropout, Dense
from keras.constraints import max_norm
from keras.layers.normalization import BatchNormalization


class Setup(object):
    """for dependency parsing with form, lemma, upostag, feats, and deprel"""
    __slots__ = 'idx2tran', 'form2idx', 'lemm2idx', 'upos2idx', 'drel2idx', \
                'feat2idx', 'form_emb', 'lemm_emb', 'x', 'y'

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    @staticmethod
    def cons(sents, form_w2v, lemm_w2v=None, proj=False):
        """[Sent], gensim.models.keyedvectors.KeyedVectors -> Setup"""
        specials = Sent.dumb, Sent.root, Sent.obsc
        # form_emb form2idx
        pad = [s for s in specials if s not in form_w2v.vocab]
        voc, dim = form_w2v.syn0.shape
        form_emb = np.zeros((voc + len(pad), dim), np.float32)
        form_emb[:len(form_w2v.index2word)] = form_w2v.syn0
        form2idx = {form: idx for idx, form in enumerate(form_w2v.index2word)}
        del form_w2v
        for form in pad:
            form2idx[form] = len(form2idx)
        # lemm_emb lemm2idx
        if lemm_w2v:
            pad = [s for s in specials if s not in lemm_w2v.vocab]
            voc, dim = lemm_w2v.syn0.shape
            lemm_emb = np.zeros((voc + len(pad), dim), np.float32)
            lemm_emb[:len(lemm_w2v.index2word)] = lemm_w2v.syn0
            lemm2idx = {lemm: idx for idx, lemm in enumerate(lemm_w2v.index2word)}
            del lemm_w2v
        else:
            pad = specials
            lemm_emb = None
            lemm2idx = {}
        for lemm in pad:
            lemm2idx[lemm] = len(lemm2idx)
        # upos2idx feat2idx drel2idx idx2tran
        upos2idx = {Sent.dumb: 0, Sent.root: 1, 'X': 2}
        feat2idx = {Sent.dumb: 0, Sent.root: 1}
        drel2idx = {Sent.dumb: 0}
        idx2tran = [('shift', None)]
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
        if not proj:
            idx2tran.append(('swap', None))
        # x y
        self = Setup(
            form2idx=form2idx, form_emb=form_emb,
            lemm2idx=lemm2idx, lemm_emb=lemm_emb,
            upos2idx=upos2idx,
            drel2idx=drel2idx,
            feat2idx=feat2idx,
            idx2tran=idx2tran)
        tran2idx = {}
        for idx, tran in enumerate(idx2tran):
            hotv = np.zeros(len(idx2tran), np.float32)
            hotv[idx] = 1.0
            tran2idx[tran] = hotv
        data = [], [], [], [], [], []
        form_append, lemm_append, upos_append, drel_append, feat_append, \
            tran_append, = [d.append for d in data]
        for sent in sents:
            oracle = Oracle.cons(sent, proj=proj)
            config = Config.cons(sent)
            while not config.is_terminal():
                tran = oracle.predict(config)
                if not config.doable(tran[0]):
                    # this happends on a non-proj sent with proj setting
                    break
                feat = self.feature(config)
                form_append(feat[0])
                lemm_append(feat[1])
                upos_append(feat[2])
                drel_append(feat[3])
                feat_append(feat[4])
                tran_append(tran2idx[tran])
                getattr(config, tran[0])(tran[1])
        self.x = [np.concatenate(d) for d in data[:-1]]
        self.y = np.array(data[-1], np.float32)
        return self

    @staticmethod
    def make(train_conllu, form_w2v, lemm_w2v=None, binary=True, proj=False):
        """-> Setup; from files"""
        return Setup.cons(
            conllu.load(train_conllu),  proj=proj,
            form_w2v=KeyedVectors.load_word2vec_format(form_w2v, binary=binary),
            lemm_w2v=KeyedVectors.load_word2vec_format(lemm_w2v, binary=binary)
            if lemm_w2v else None)

    def model(self,
              upos_embed_dim=12,
              drel_embed_dim=16,
              hidden_units=256,
              hidden_layers=2,
              hidden_bn=False,
              
              activation='relu',
              init='he_uniform'

              embed_const='unit_norm',
              embed_dropout=0.25,
              hidden_const=None,
              hidden_dropout=0.25,
              output_const=None,
              
              optimizer='adamax'):
        """-> keras.models.Model"""
        assert 0 <= upos_embed_dim
        assert 0 <= drel_embed_dim
        assert 0 <= embed_dropout < 1
        assert 0 <= hidden_layers
        assert 0 < hidden_units or 0 == hidden_layers
        assert 0 <= hidden_dropout < 1

        def const(x):
            try:
                x = float(x)
                assert 0 < x
                x = max_norm(x)
            except (TypeError, ValueError):
                if isinstance(x, str) and "none" == x.lower():
                    x = None
            return x

        embed_const = const(embed_const)
        hidden_const = const(hidden_const)
        output_const = const(output_const)
        
        form = Input(name="form", shape=self.x[0].shape[1:], dtype=np.uint16)
        lemm = Input(name="lemm", shape=self.x[1].shape[1:], dtype=np.uint16)
        upos = Input(name="upos", shape=self.x[2].shape[1:], dtype=np.uint8)
        drel = Input(name="drel", shape=self.x[3].shape[1:], dtype=np.uint8)
        feat = Input(name="feat", shape=self.x[4].shape[1:], dtype=np.float32)
        i = [form, lemm, upos, drel, feat]
        form = Embedding(
            input_dim=len(self.form2idx),
            output_dim=self.form_emb.shape[-1],
            weights=[self.form_emb],
            embeddings_constraint=embed_const,
            name="form_embed")(form)
        lemm = Embedding(
            input_dim=len(self.lemm2idx),
            output_dim=self.lemm_emb.shape[-1],
            weights=[self.lemm_emb],
            embeddings_constraint=embed_const,
            name="lemm_embed")(lemm) if self.lemm_emb is not None else form
        upos = Embedding(
            input_dim=len(self.upos2idx),
            output_dim=upos_embed_dim,
            embeddings_initializer=init,
            embeddings_constraint=embed_const,
            name="upos_embed")(upos)
        drel = Embedding(
            input_dim=len(self.drel2idx),
            output_dim=drel_embed_dim,
            embeddings_initializer=init,
            embeddings_constraint=embed_const,
            name="drel_embed")(drel)
        form = Flatten(name="form_flat")(form)
        lemm = Flatten(name="lemm_flat")(lemm)
        upos = Flatten(name="upos_flat")(upos)
        drel = Flatten(name="drel_flat")(drel)
        if embed_dropout:
            form = Dropout(name="form_dropout", rate=embed_dropout)(form)
            lemm = Dropout(name="lemm_dropout", rate=embed_dropout)(lemm)
            upos = Dropout(name="upos_dropout", rate=embed_dropout)(upos)
            drel = Dropout(name="drel_dropout", rate=embed_dropout)(drel)
        o = Concatenate(name="concat")(
            [form, lemm, upos, drel, feat] if self.lemm_emb is not None else
            [form, upos, drel, feat])
        for hid in range(hidden_layers):
            o = Dense(
                units=hidden_units,
                activation=activation,
                bias_initializer='ones' if 'relu' == activation else 'zeros',
                kernel_initializer=init,
                kernel_constraint=hidden_const,
                name="hidden{}".format(1 + hid))(o)
            if hidden_dropout:
                o = Dropout(name="hidden{}_dropout".format(1 + hid), rate=hidden_dropout)(o)
            elif hidden_bn:
                o = BatchNormalization(name="hidden{}_bn".format(1 + hid))(o)
        o = Dense(
            units=len(self.idx2tran),
            activation='softmax',
            kernel_initializer=init,
            kernel_constraint=output_const,
            name="output")(o)
        m = Model(i, o, name="darc")
        m.compile(optimizer, 'categorical_crossentropy')
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

    def feature(self, config):
        """-> [numpy.ndarray] :as form, upos, drel, feat

        assert form.shape == upos.shape == (18, )

        assert drel.shape == (16, )

        assert feat.shape == (18 * len(self.feat2idx), )

        """
        # 18 features (Chen & Manning 2014)
        #  0: s0       1: s1       2: s0l1     3: s1l1     4: s0r1     5: s1r1
        #  6: s0l0     7: s1l0     8: s0r0     9: s1r0
        # 10: s0l0l1  11: s1l0l1  12: s0r0r1  13: s1r0r1
        # 14: s2      15: i0      16: i1      17: i2
        i, s, g = config.input, config.stack, config.graph
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
        # node 0 in each sent is dumb
        if 1 <= len(s):
            x[0] = s[-1]  # s0
            y = g[x[0]]
            if y:
                if 2 <= len(y):
                    x[2] = y[1]  # s0l1
                    x[4] = y[-2]  # s0r1
                x[6] = y[0]  # s0l0
                x[8] = y[-1]  # s0r0
                y = g[x[6]]
                if y:
                    x[10] = y[0]  # s0l0l1
                y = g[x[8]]
                if y:
                    x[12] = y[-1]  # s0r0r1
            if 2 <= len(s):
                x[1] = s[-2]  # s1
                y = g[x[1]]
                if y:
                    if 2 <= len(y):
                        x[3] = y[1]  # s1l1
                        x[5] = y[-2]  # s1r1
                    x[7] = y[0]  # s1l0
                    x[9] = y[-1]  # s1r0
                    y = g[x[7]]
                    if y:
                        x[11] = y[0]  # s1l0l1
                    y = g[x[9]]
                    if y:
                        x[13] = y[-1]  # s1r0r1
                if 3 <= len(s):
                    x[14] = s[-3]  # s2
        if 1 <= len(i):
            x[15] = i[-1]  # i0
            if 2 <= len(i):
                x[16] = i[-2]  # i1
                if 3 <= len(i):
                    x[17] = i[-3]  # i2
        # form lemm upos
        form2idx = self.form2idx.get
        lemm2idx = self.lemm2idx.get
        upos2idx = self.upos2idx.get
        form_unk = form2idx(Sent.obsc)
        lemm_unk = lemm2idx(Sent.obsc)
        upos_unk = upos2idx('X')  # upos is never _
        form = config.sent.form
        lemm = config.sent.lemma
        upos = config.sent.upostag
        form = np.fromiter((form2idx(form[i], form_unk) for i in x), np.uint16)
        lemm = np.fromiter((lemm2idx(lemm[i], lemm_unk) for i in x), np.uint16)
        upos = np.fromiter((upos2idx(upos[i], upos_unk) for i in x), np.uint8)
        # drel
        drel2idx = self.drel2idx
        drel = config.deprel
        drel = np.fromiter((drel2idx[drel[i]] for i in x[2:]), np.uint8)
        # feats
        feats = config.sent.feats
        feats = [feats[i] for i in x]
        # special treatments for root
        if 2 <= len(s) and 0 == s[-2]:
            root = Sent.root
            # s1 at idx 1 is root
            form[1] = form2idx(root)
            lemm[1] = lemm2idx(root)
            upos[1] = upos2idx(root)
            feats[1] = root
        # set-valued feat (Alberti et al. 2015)
        feat2idx = self.feat2idx
        feat = np.zeros((len(feats), len(feat2idx)), np.float32)
        for idx, fts in enumerate(feats):
            for ft in fts.split("|"):
                try:
                    feat[idx, feat2idx[ft]] = 1.0
                except KeyError:
                    pass
        form.shape = lemm.shape = upos.shape = drel.shape = feat.shape = 1, -1
        return [form, lemm, upos, drel, feat]  # model.predict takes list

    def bean(self, with_data=True):
        """-> dict"""
        return {attr: getattr(self, attr) for attr in
                (Setup.__slots__ if with_data else Setup.__slots__[:-4])}

    def save(self, file, with_data=True):
        """as npy file"""
        np.save(file, self.bean(with_data))

    @staticmethod
    def load(file):
        """-> Setup"""
        return Setup(**np.load(file).item())
    

# lang, proj = 'kk', False
# import src_ud2 as ud2
# setup = Setup.make(
#     ud2.path(lang, 'train'), proj=proj,
#     form_w2v="./lab/embed/{}-form.w2v".format(lang),
#     lemm_w2v="./lab/embed/{}-lemm.w2v".format(lang),
#     binary=False)
# from keras.utils import plot_model
# model = setup.model()
# plot_model(model, to_file="./lab/model.png")
