from itertools import repeat
from transition import Config, Oracle
from conllu import Sent, load
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dropout, Dense
from keras.constraints import max_norm


class Setup(object):
    """for dependency parsing with form, lemma, upostag, feats, and deprel"""
    __slots__ = 'idx2tran', 'form2idx', 'lemm2idx', 'upos2idx', 'drel2idx', \
                'feat2idx', 'form_emb', 'lemm_emb', 'x', 'y'

    dumb_form, root_form, obsc_form = Sent.dumb[1], "</s>", "_"
    dumb_lemm, root_lemm, obsc_lemm = Sent.dumb[2], "</s>", "_"
    dumb_upos, root_upos = Sent.dumb[3], "ROOT"
    dumb_feat, root_feat = Sent.dumb[5], "Root=Yes"
    dumb_drel = Sent.dumb[7]

    def __init__(self, **kwargs):
        super().__init__()
        for attr, val in kwargs.items():
            setattr(self, attr, val)

    @staticmethod
    def cons(sents, form_w2v, lemm_w2v, proj=False):
        """[Sent], gensim.models.keyedvectors.KeyedVectors -> Setup"""
        # form_emb form2idx
        specials = Setup.dumb_form, Setup.root_form, Setup.obsc_form
        pad = 0
        for form in specials:
            if form not in form_w2v.vocab:
                pad += 1
        voc, dim = form_w2v.syn0.shape
        form_emb = np.zeros((pad + voc, dim), np.float32)
        form_emb[:len(form_w2v.index2word)] = form_w2v.syn0
        form2idx = {form: idx for idx, form in enumerate(form_w2v.index2word)}
        del form_w2v
        for form in specials:
            if form not in form2idx:
                form2idx[form] = len(form2idx)
        # lemm_emb lemm2idx
        specials = Setup.dumb_lemm, Setup.root_lemm, Setup.obsc_lemm
        pad = 0
        for lemm in specials:
            if lemm not in lemm_w2v.vocab:
                pad += 1
        voc, dim = lemm_w2v.syn0.shape
        lemm_emb = np.zeros((pad + voc, dim), np.float32)
        lemm_emb[:len(lemm_w2v.index2word)] = lemm_w2v.syn0
        lemm2idx = {lemm: idx for idx, lemm in enumerate(lemm_w2v.index2word)}
        del lemm_w2v
        for lemm in specials:
            if lemm not in lemm2idx:
                lemm2idx[lemm] = len(lemm2idx)
        # upos2idx feat2idx drel2idx idx2tran
        upos2idx = {Setup.dumb_upos: 0, Setup.root_upos: 1}
        feat2idx = {Setup.dumb_feat: 0, Setup.root_feat: 1}
        drel2idx = {}
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
        drel2idx[Setup.dumb_drel] = len(drel2idx)
        if not proj:
            idx2tran.append(('swap', None))
        # x y
        self = Setup(
            form2idx=form2idx,
            form_emb=form_emb,
            lemm2idx=lemm2idx,
            lemm_emb=lemm_emb,
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
    def make(train_conllu, form_w2v, lemm_w2v, binary=True, proj=False):
        """-> Setup; from files"""
        return Setup.cons(
            load(train_conllu),
            KeyedVectors.load_word2vec_format(form_w2v, binary=binary),
            KeyedVectors.load_word2vec_format(lemm_w2v, binary=binary),
            proj=proj)

    def model(self,
              upos_emb_dim=12,
              drel_emb_dim=16,
              hidden_units=200,
              emb_init='uniform',
              dense_init='orthogonal',
              emb_const='unit_norm',
              dense_const=None,
              emb_dropout=0.25,
              dense_dropout=0.0,
              activation='tanh',
              optimizer='adamax'):
        """-> keras.models.Model"""
        try:
            float(emb_const)
        except (TypeError, ValueError):
            pass
        else:
            emb_const = max_norm(emb_const)
        try:
            float(dense_const)
        except (TypeError, ValueError):
            pass
        else:
            dense_const = max_norm(dense_const)
        num_node = 18
        form = Input(name="form", shape=(num_node, ), dtype=np.uint16)
        lemm = Input(name="lemm", shape=(num_node, ), dtype=np.uint16)
        upos = Input(name="upos", shape=(num_node, ), dtype=np.uint8)
        drel = Input(name="drel", shape=(num_node - 2, ), dtype=np.uint8)
        feat = Input(name="feat", shape=(num_node * len(self.feat2idx), ))
        i = [form, lemm, upos, drel, feat]
        form = Embedding(
            input_dim=len(self.form2idx),
            input_length=num_node,
            output_dim=self.form_emb.shape[-1],
            weights=[self.form_emb],
            embeddings_constraint=emb_const,
            name="form_emb")(form)
        lemm = Embedding(
            input_dim=len(self.lemm2idx),
            input_length=num_node,
            output_dim=self.lemm_emb.shape[-1],
            weights=[self.lemm_emb],
            embeddings_constraint=emb_const,
            name="lemm_emb")(lemm)
        upos = Embedding(
            input_dim=len(self.upos2idx),
            input_length=num_node,
            output_dim=upos_emb_dim,
            embeddings_initializer=emb_init,
            embeddings_constraint=emb_const,
            name="upos_emb")(upos)
        drel = Embedding(
            input_dim=len(self.drel2idx),
            input_length=num_node - 2,
            output_dim=drel_emb_dim,
            embeddings_initializer=emb_init,
            embeddings_constraint=emb_const,
            name="drel_emb")(drel)
        form = Flatten(name="form_flat")(form)
        lemm = Flatten(name="lemm_flat")(lemm)
        upos = Flatten(name="upos_flat")(upos)
        drel = Flatten(name="drel_flat")(drel)
        if emb_dropout:
            form = Dropout(name="form_dropout", rate=emb_dropout)(form)
            lemm = Dropout(name="lemm_dropout", rate=emb_dropout)(lemm)
            upos = Dropout(name="upos_dropout", rate=emb_dropout)(upos)
            drel = Dropout(name="drel_dropout", rate=emb_dropout)(drel)
        o = Concatenate(name="inputs")([form, lemm, upos, drel, feat])
        o = Dense(
            units=hidden_units,
            activation=activation,
            kernel_initializer=dense_init,
            kernel_constraint=dense_const,
            name="hidden")(o)
        if dense_dropout:
            o = Dropout(name="hidden_dropout", rate=dense_dropout)(o)
        o = Dense(
            units=len(self.idx2tran),
            activation='softmax',
            kernel_initializer=dense_init,
            kernel_constraint=dense_const,
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
        num_node = 18
        # 18 features (Chen & Manning 2014)
        #  0: s0       1: s1       2: s0l1     3: s1l1     4: s0r1     5: s1r1
        #  6: s0l0     7: s1l0     8: s0r0     9: s1r0
        # 10: s0l0l1  11: s1l0l1  12: s0r0r1  13: s1r0r1
        # 14: s2      15: i0      16: i1      17: i2
        i, s, g = config.input, config.stack, config.graph
        x = list(repeat(0, num_node))  # node 0 in each sent is dumb
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
        form_unk = form2idx(self.obsc_form)
        lemm_unk = lemm2idx(self.obsc_lemm)
        upos_unk = upos2idx(self.dumb_upos)
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
            # s1 at idx 1 is root
            form[1] = form2idx(self.root_form)
            lemm[1] = lemm2idx(self.root_lemm)
            upos[1] = upos2idx(self.root_upos)
            feats[1] = self.root_feat
        # set-valued feat (Alberti et al. 2015)
        feat2idx = self.feat2idx
        num_feat = len(feat2idx)
        feat = np.zeros(num_node * num_feat, np.float32)
        for idx, fts in enumerate(feats):
            for ft in fts.split("|"):
                try:
                    feat[num_feat * idx + feat2idx[ft]] = 1.0
                except KeyError:
                    pass
        form.shape = lemm.shape = upos.shape = drel.shape = feat.shape = 1, -1
        return [form, lemm, upos, drel, feat]  # model.predict takes list

    def bean(self, with_data=True):
        """-> dict"""
        attrs = Setup.__slots__ if with_data else Setup.__slots__[:-4]
        return {attr: getattr(self, attr) for attr in attrs}

    def save(self, file, with_data=True):
        """as npy file"""
        np.save(file, self.bean(with_data))

    @staticmethod
    def load(file):
        """-> Setup"""
        return Setup(**np.load(file).item())


# lang, proj = 'kk', False
# from ud2 import path
# setup = Setup.make(
#     path(lang, 'train'),
#     "./embed/{}-form.w2v".format(lang),
#     "./embed/{}-lemm.w2v".format(lang),
#     binary=False,
#     proj=proj)
# from keras.utils import plot_model
# model = setup.model()
# plot_model(model, to_file=".tmp/model.png")
