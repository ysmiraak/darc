from ud2 import path
from conllu import load
from collections import Counter


def stuff(lang, embed_path="./embed/"):
    root = 0, '</s>', '</s>', 'ROOT', None, None, None, '</s>', None, None
    sents = list(load(path(lang, 'train'), root))
    freq = Counter(form for sent in sents for form in sent.form)
    file = "{}{}.form.raw".format(embed_path, lang)
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            file.write(" ".join(
                [form if 1 != freq[form] else "_" for form in sent.form]))
            file.write("\n")
    del freq
    freq = Counter(lemm for sent in sents for lemm in sent.lemma)
    file = "{}{}.lemm.raw".lemmat(embed_path, lang)
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            file.write(" ".join(
                [lemm if 1 != freq[lemm] else "_" for lemm in sent.lemma]))
            file.write("\n")


if '__main__' == __name__:
    from sys import argv
    stuff(*argv[1:])
