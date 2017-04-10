from ud2 import ud_path, treebanks
from conllu import load
from collections import Counter
from gensim.models.word2vec import Word2Vec


def pretrain(lang, suffix="", window=8, negative=8, sample=0.1):
    sents = list(
        load(
            ud_path + "UD_{}/{}-ud-train.conllu".format(treebanks[lang], lang),
            (0, '</s>', '</s>', 'ROOT', None, None, None, '</s>', None, None),
        ))
    freq = Counter(form for sent in sents for form in sent.form)
    Word2Vec(
        sentences=[[form if 1 != freq[form] else "_" for form in sent.form]
                   for sent in sents],
        size=32,
        iter=16,
        window=window,
        min_count=2,
        sg=1,
        hs=0,
        negative=negative,
        sample=sample,
    ).wv.save_word2vec_format("./embed/{}-form{}.w2v".format(lang, suffix))
    freq = Counter(lemm for sent in sents for lemm in sent.lemma)
    Word2Vec(
        sentences=[[lemm if 1 != freq[lemm] else "_" for lemm in sent.lemma]
                   for sent in sents],
        size=32,
        iter=16,
        window=window,
        min_count=2,
        sg=1,
        hs=0,
        negative=negative,
        sample=sample,
    ).wv.save_word2vec_format("./embed/{}-lemm{}.w2v".format(lang, suffix))
