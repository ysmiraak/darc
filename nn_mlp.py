from itertools import repeat
from transition import Config, Oracle
from conllu import Word, load
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from gensim.models.keyedvectors import KeyedVectors

# import h5py
# import json


class Setup(object):
    """sents: [Sent], w2v: gensim.models.keyedvectors.KeyedVectors"""
    __slots__ = 'form2idx', 'upos2idx', 'feat2idx', 'idx2tran', \
                'form_emb', 'x', 'y'

    unknown = Word(None)

    def __init__(self, sents, w2v, labeled=True, projective=False):
        self.labeled = labeled
        self.projective = projective
        # form_emb form2idx
        form_emb = [np.zeros(50, 'float32')]
        form2idx = {Setup.unknown.form: 0}
        form_emb_append = form_emb.append
        for form in w2v.index2word:
            form_emb_append(w2v[form])
            form2idx[form] = len(form2idx)
        self.form_emb = np.array(form_emb, 'float32')
        self.form2idx = form2idx
        # upos2idx feat2idx idx2tran
        upos2idx = {Setup.unknown.upostag: 0, '</s>': 1}  # 1:root
        feat2idx = {}  # TODO: normalize with {None: 0}
        rels = set() if labeled else [None]
        if not hasattr(sents, '__len__'):
            sents = list(sents)
        for sent in sents:
            for word in sent.iter_words():
                if word.upostag not in upos2idx:
                    upos2idx[word.upostag] = len(upos2idx)
                for feat in word.feats:
                    if feat not in feat2idx:
                        feat2idx[feat] = len(feat2idx)
                if labeled:
                    rels.add(word.deprel)
        self.upos2idx = upos2idx
        self.feat2idx = feat2idx
        self.idx2tran = [('shift', None)]
        self.idx2tran.extend(('right', rel) for rel in rels)
        self.idx2tran.extend(('left', rel) for rel in rels)
        if not projective:
            self.idx2tran.append(('swap', None))
        # x y
        tran2idx = {}
        for idx, tran in enumerate(self.idx2tran):
            hotv = np.zeros(len(self.idx2tran), 'float32')
            hotv[idx] = 1.0
            tran2idx[tran] = hotv
        yx = [], [], [], []
        tran_append = yx[0].append
        form_append = yx[1].append
        upos_append = yx[2].append
        feat_append = yx[3].append
        for sent in sents:
            oracle = Oracle(sent, proj=projective)
            config = Config(sent)
            while not config.is_terminal():
                tran = oracle.predict(config)
                if 'shift' == tran[0] and not config.input:
                    # this happends on a non-proj sent with proj setting
                    break
                feat = Setup.feature(self, config)
                form_append(feat[0])
                upos_append(feat[1])
                feat_append(feat[2])
                tran_append(tran2idx[tran])
                getattr(config, tran[0])(tran[1])
        self.x = {
            'form': np.concatenate(yx[1]),
            'upos': np.concatenate(yx[2]),
            'feat': np.concatenate(yx[3])
        }
        self.y = np.array(yx[0], 'float32')

    @staticmethod
    def build(train_conllu, embedding_txt):
        """-> Setup; build from files"""
        return Setup(
            load(train_conllu),
            KeyedVectors.load_word2vec_format(embedding_txt))

    def model(self, upos_emb_dim=10, hidden_units=200, optimizer='sgd'):
        """-> keras.models.Model

        feature: Feature

        w2v: gensim.models.keyedvectors.KeyedVectors

        """
        form = Input(name='form', shape=(18, ), dtype='int32')
        upos = Input(name='upos', shape=(18, ), dtype='int32')
        feat = Input(name='feat', shape=(18 * len(self.feat2idx), ))
        i = [form, upos, feat]
        form = Embedding(
            name='form_emb',
            input_dim=len(self.form2idx),
            output_dim=50,
            input_length=18)(form)
        upos = Embedding(
            name='upos_emb',
            input_dim=len(self.upos2idx),
            output_dim=upos_emb_dim,
            input_length=18)(upos)
        # TODO: try unit_norm constraint for embedding layers
        form = Flatten(name='form_flat')(form)
        upos = Flatten(name='upos_flat')(upos)
        o = Concatenate(name='inputs')([form, upos, feat])
        o = Dense(name='hidden', units=hidden_units, activation='tanh')(o)
        o = Dense(
            name='output', units=len(self.idx2tran), activation='softmax')(o)
        # TODO: add regularization
        m = Model(i, o, 'darc')
        m.compile(
            optimizer=optimizer,
            # sgd rmsprop adagrad adadelta adam adamax nadam
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        m.get_layer('form_emb').set_weights([self.form_emb])
        return m

    def train(self, model, *args, **kwargs):
        """mutates model by calling keras.models.Model.fit"""
        model.fit(self.x, self.y, *args, **kwargs)

    def parse(self, model, sent):
        """mutates sent"""
        config = Config(sent)
        while not config.is_terminal():
            if 2 > len(config.stack):
                config.shift()
                continue
            prob = model.predict(Setup.feature(self, config), 1).flatten()
            rank = reversed(prob.argsort())
            good = False
            for r in rank:
                act, arg = self.idx2tran[r]
                if config.doable(act):
                    getattr(config, act)(arg)
                    good = True
                    break
            if not good:
                print("WARNING!!!! FAILED TO PARSE:",
                      " ".join([w.form for w in sent]))
                return

    def feature(self, config):
        """Config -> [numpy.ndarray]

        form, upos, feat = self[config]

        assert form.shape == upos.shape == (18, )

        assert feat.shape == (18 * len(self.feat2idx), )

        """
        x = list(repeat(-1, 18))
        #  0: s2
        #  1: s1l1l1  2: s1l1   3: s1l2   4: s1   5: s1r2   6: s1r1   7: s1r1r1
        #  8: s0l1l1  9: s0l1  10: s0l2  11: s0  12: s0r2  13: s0r1  14: s0r1r1
        # 15: i0     16: i1    17: i2
        if 1 <= len(config.stack):
            x[11] = config.stack[-1]  # s0
            y = config.graph[x[11]]
            if y:
                if 2 <= len(y):
                    x[10] = y[1]  # s0l2
                    x[12] = y[-2]  # s0r2
                x[9] = y[0]  # s0l1
                x[13] = y[-1]  # s0r1
                y = config.graph[x[9]]
                if y:
                    x[8] = y[0]  # s0l1l1
                y = config.graph[x[13]]
                if y:
                    x[14] = y[-1]  # s0r1r1
            if 2 <= len(config.stack):
                x[4] = config.stack[-2]  # s1
                y = config.graph[x[4]]
                if y:
                    if 2 <= len(y):
                        x[3] = y[1]  # s1l2
                        x[5] = y[-2]  # s1r2
                    x[2] = y[0]  # s1l1
                    x[6] = y[-1]  # s1r1
                    y = config.graph[x[2]]
                    if y:
                        x[1] = y[0]  # s1l1l1
                    y = config.graph[x[6]]
                    if y:
                        x[7] = y[-1]  # s1l1r1
                if 3 <= len(config.stack):
                    x[0] = config.stack[-3]  # s2
        if 1 <= len(config.input):
            x[15] = config.input[-1]  # i0
            if 2 <= len(config.input):
                x[16] = config.input[-2]  # i1
                if 3 <= len(config.input):
                    x[17] = config.input[-3]  # i2
        # 18 features (Chen & Manning 2014)
        x = [config.words[x] if -1 != x else Setup.unknown for x in x]
        # set-valued feat (Alberti et al. 2015)
        feat = []
        for w in x:
            ft = np.zeros(len(self.feat2idx), 'float32')
            for i in [self.feat2idx[f] for f in w.feats if f in self.feat2idx]:
                ft[i] = 1.0
            # TODO: try normalizing ft
            # for i in [self.feat2idx.get(feat, 0) for feat in w.feats]:
            #     ft[i] = 1.0
            # ft[0] = 1.0
            feat.append(ft)
        feat = np.concatenate(feat)
        # TODO: add valency feature drel
        form = np.array([self.form2idx.get(w.form, 0) for w in x], 'int32')
        upos = np.array([self.upos2idx.get(w.upostag, 0) for w in x], 'int32')
        form.shape = (1, 18)
        upos.shape = (1, 18)
        feat.shape = (1, feat.size)
        return [form, upos, feat]

    def save():
        pass

    def load():
        pass


# ud_path = "/data/ud-treebanks-conll2017/UD_Ancient_Greek-PROIEL/"
# wv_path = ("/data/udpipe-ud-2.0-conll17-170315-supplementary-data/"
#            "ud-2.0-baselinemodel-train-embeddings/")
# setup = Setup.build(ud_path + "grc_proiel-ud-train.conllu",
#                     wv_path + "grc_proiel.skip.forms.50.vectors")
