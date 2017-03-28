from itertools import repeat
from transition import Config, Oracle
from conllu import Word, load, validate, write
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from gensim.models.keyedvectors import KeyedVectors


class Setup(object):
    """sents: [Sent], w2v: gensim.models.keyedvectors.KeyedVectors"""
    __slots__ = 'form2idx', 'upos2idx', 'feat2idx', 'drel2idx', 'idx2tran', \
                'form_emb', 'x', 'y'

    unknown = Word(None)

    def __init__(self, sents, w2v, labeled=True, projective=False):
        super().__init__()
        if not sents:
            return
        # form_emb form2idx
        form_emb = [np.zeros(50, np.float32)]
        form2idx = {Setup.unknown.form: 0}
        form_emb_append = form_emb.append
        for form in w2v.index2word:
            form_emb_append(w2v[form])
            form2idx[form] = len(form2idx)
        self.form_emb = np.array(form_emb, np.float32)
        self.form2idx = form2idx
        # upos2idx feat2idx idx2tran
        upos2idx = {Setup.unknown.upostag: 0, 'ROOT': 1}  # 1:root
        feat2idx = {}
        rels = set() if labeled else [None]
        if not hasattr(sents, '__len__'):
            sents = list(sents)
        for sent in sents:
            for word in sent.iter_words():
                if word.upostag not in upos2idx:
                    upos2idx[word.upostag] = len(upos2idx)
                for feat in word.feats.split("|"):
                    if feat not in feat2idx:
                        feat2idx[feat] = len(feat2idx)
                if labeled:
                    rels.add(word.deprel)
        self.upos2idx = upos2idx
        self.feat2idx = feat2idx
        self.drel2idx = {drel: idx for idx, drel in enumerate(rels)}
        self.idx2tran = [('shift', None)]
        self.idx2tran.extend(('right', rel) for rel in rels)
        self.idx2tran.extend(('left', rel) for rel in rels)
        if not projective:
            self.idx2tran.append(('swap', None))
        # x y
        tran2idx = {}
        for idx, tran in enumerate(self.idx2tran):
            hotv = np.zeros(len(self.idx2tran), np.float32)
            hotv[idx] = 1.0
            tran2idx[tran] = hotv
        data = [], [], [], [], []
        tran_append, form_append, upos_append, feat_append, drel_append \
            = [d.append for d in data]
        for sent in sents:
            oracle = Oracle(sent, projective=projective, labeled=labeled)
            config = Config(sent)
            while not config.is_terminal():
                tran = oracle.predict(config)
                if 'shift' == tran[0] and not config.input:
                    # this happends on a non-proj sent with proj setting
                    break
                feat = Setup.feature(self, config)
                tran_append(tran2idx[tran])
                form_append(feat[0])
                upos_append(feat[1])
                feat_append(feat[2])
                drel_append(feat[3])
                getattr(config, tran[0])(tran[1], False)
        self.y = np.array(data[0], np.float32)
        self.x = [np.concatenate(d) for d in data[1:]]

    @staticmethod
    def build(train_conllu, embedding_txt, labeled=True, projective=False):
        """-> Setup; build from files"""
        return Setup(
            load(train_conllu),
            KeyedVectors.load_word2vec_format(embedding_txt),
            labeled=labeled,
            projective=projective)

    def model(self, upos_emb_dim=10, hidden_units=200, optimizer='adamax'):
        """-> keras.models.Model

        feature: Feature

        w2v: gensim.models.keyedvectors.KeyedVectors

        """
        form = Input(name='form', shape=(18, ), dtype=np.uint16)
        upos = Input(name='upos', shape=(18, ), dtype=np.uint8)
        feat = Input(name='feat', shape=(18 * len(self.feat2idx), ))
        drel = Input(name='drel', shape=(2 * len(self.drel2idx), ))
        i = [form, upos, feat, drel]
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
        form = Flatten(name='form_flat')(form)
        upos = Flatten(name='upos_flat')(upos)
        o = Concatenate(name='inputs')([form, upos, feat, drel])
        o = Dense(name='hidden', units=hidden_units, activation='tanh')(o)
        o = Dense(
            name='output', units=len(self.idx2tran), activation='softmax')(o)
        m = Model(i, o, 'darc')
        m.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        m.get_layer('form_emb').set_weights([self.form_emb.copy()])
        # copy necessary ????
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
            prob = model.predict(Setup.feature(self, config), 1).ravel()
            good = False
            for r in prob.argsort()[::-1]:
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
        """Config -> [numpy.ndarray] :as form, upos, feat

        assert form.shape == upos.shape == (18, )

        assert feat.shape == (18 * len(self.feat2idx), )

        """
        w, i, s, g = config.words, config.input, config.stack, config.graph
        x = list(repeat(-1, 18))
        #  0: s2
        #  1: s1l0l1  2: s1l0   3: s1l1   4: s1   5: s1r1   6: s1r0   7: s1r0r1
        #  8: s0l0l1  9: s0l0  10: s0l1  11: s0  12: s0r1  13: s0r0  14: s0r0r1
        # 15: i0     16: i1    17: i2
        if 1 <= len(s):
            x[11] = s[-1]  # s0
            y = g[x[11]]
            if y:
                if 2 <= len(y):
                    x[10] = y[1]  # s0l1
                    x[12] = y[-2]  # s0r1
                x[9] = y[0]  # s0l0
                x[13] = y[-1]  # s0r0
                y = g[x[9]]
                if y:
                    x[8] = y[0]  # s0l0l1
                y = g[x[13]]
                if y:
                    x[14] = y[-1]  # s0r0r1
            if 2 <= len(s):
                x[4] = s[-2]  # s1
                y = g[x[4]]
                if y:
                    if 2 <= len(y):
                        x[3] = y[1]  # s1l1
                        x[5] = y[-2]  # s1r1
                    x[2] = y[0]  # s1l0
                    x[6] = y[-1]  # s1r0
                    y = g[x[2]]
                    if y:
                        x[1] = y[0]  # s1l0l1
                    y = g[x[6]]
                    if y:
                        x[7] = y[-1]  # s1r0r1
                if 3 <= len(s):
                    x[0] = s[-3]  # s2
        if 1 <= len(i):
            x[15] = i[-1]  # i0
            if 2 <= len(i):
                x[16] = i[-2]  # i1
                if 3 <= len(i):
                    x[17] = i[-3]  # i2
        # 18 features (Chen & Manning 2014)
        words = [w[i] if -1 != i else Setup.unknown for i in x]
        # set-valued feat (Alberti et al. 2015)
        feats = []
        for word in words:
            featv = np.zeros(len(self.feat2idx), np.float32)
            for feat in word.feats.split("|"):
                try:
                    featv[self.feat2idx[feat]] = 1.0
                except KeyError:
                    continue
            feats.append(featv)
        # add valency feature drel
        nr_drel = len(self.drel2idx)
        drel = np.zeros(2 * nr_drel, np.float32)
        for i in g[x[4]]:
            try:
                drel[self.drel2idx[w[i].deprel]] = 1.0
            except KeyError:
                continue
        for i in g[x[11]]:
            try:
                drel[nr_drel + self.drel2idx[w[i].deprel]] = 1.0
            except KeyError:
                continue
        return [
            np.array([self.form2idx.get(word.form, 0) for word in words],
                     np.uint16).reshape(1, 18),
            np.array([self.upos2idx.get(word.upostag, 0) for word in words],
                     np.uint8).reshape(1, 18),
            np.concatenate(feats).reshape(1, -1),
            drel.reshape(1, -1)
        ]  # model.predict takes list not tuple

    def save(self, file):
        """as npy file"""
        np.save(file, {a: getattr(self, a) for a in Setup.__slots__})

    @staticmethod
    def load(file):
        """-> Setup"""
        setup = Setup(None, None)
        data = np.load(file).item()
        for a in Setup.__slots__:
            setattr(setup, a, data[a])
        return setup


ud_path = "/data/ud-treebanks-conll2017/"
wv_path = "/data/udpipe-ud-2.0-conll17-170315-supplementary-data/" \
          "ud-2.0-baselinemodel-train-embeddings/"

setup = Setup.build(
    ud_path + "UD_Ancient_Greek-PROIEL/grc_proiel-ud-train.conllu",
    wv_path + "grc_proiel.skip.forms.50.vectors")

dev = list(load(ud_path + "/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"))

optimizer = 'adamax'
model = setup.model(optimizer=optimizer)
for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "./results/{}_valency_e{}.conllu".format(optimizer, epoch))
