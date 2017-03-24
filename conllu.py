from itertools import chain, islice


def is_pos_int(x):
    """-> bool"""
    return isinstance(x, int) and 0 < x


def is_nat_int(x):
    """-> bool"""
    return isinstance(x, int) and 0 <= x


class Fault(object):
    """like Exception but for collecting and not throwing"""
    __slots__ = 'attr', 'val', 'spec'

    def __init__(self, attr, val, spec):
        super().__init__()
        self.attr = attr
        self.val = val
        self.spec = spec

    def __repr__(self):
        return "Fault({}, {}, {})".format(self.attr, self.val, self.spec)

    def __str__(self):
        return "{} {} is not {}".format(self.attr, self.val, self.spec)


class Token(object):
    """sum type of Word and MultiWord"""
    __slots__ = 'form', 'lemma', 'upostag', 'xpostag', \
                'feats', 'head', 'deprel', 'deps', 'misc'

    def validate(self, acc=None):
        """-> [Fault]"""
        if acc is None:
            acc = []
        for a in Token.__slots__:
            if 'head' == a:
                continue
            v = getattr(self, a)
            if not isinstance(v, str):
                acc.append(Fault(a, v, 'str'))
            if not v:
                acc.append(Fault(a, v, "non-empty"))
        if "_" != self.feats \
           and any(2 != len(f.split("=")) for f in self.feats.split("|")):
            acc.append(Fault('feats', self.feats, "wellformed"))
        return acc

    def __init__(self, form="_", lemma="_", upostag="_", xpostag="_",
                 feats="_", head="_", deprel="_", deps="_", misc="_"):
        super().__init__()
        self.form, self.lemma, self.upostag, self.xpostag, \
            self.feats, self.head, self.deprel, self.deps, self.misc \
            = form, lemma, upostag, xpostag, feats, head, deprel, deps, misc

    def __eq__(self, other):
        return self is other or isinstance(other, Token) and \
            all(getattr(self, a) == getattr(other, a) for a in Token.__slots__)

    def __repr__(self):
        return ", ".join([repr(getattr(self, a)) for a in Token.__slots__])

    def __str__(self):
        return "\t".join([str(getattr(self, a)) for a in Token.__slots__])

    @staticmethod
    def parse(s):
        """str -> Word | MultiWord"""
        args = s.split("\t")
        if len(args) != 10:
            raise TypeError("expected 10-col tsv, got: {}".format(s))
        if "-" in args[0]:
            lo, hi = args[0].split("-")
            args[0] = int(hi)
            return MultiWord(int(lo), *args)
        args[0] = int(args[0])
        if "_" != args[6]:
            args[6] = int(args[6])
        return Word(*args)


class Word(Token):
    """[id | Token.__slots__]"""
    __slots__ = 'id'

    def validate(self, acc=None):
        if acc is None:
            acc = []
        if not is_pos_int(self.id):
            acc.append(Fault('id', self.id, "pos-int"))
        if not is_nat_int(self.head):
            acc.append(Fault('head', self.head, "nat-int"))
        return Token.validate(self, acc)

    def __init__(self, id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id

    def __eq__(self, other):
        return self is other or isinstance(other, Word) and \
            self.id == other.id and Token.__eq__(self, other)

    def __repr__(self):
        return "Word({}, {})".format(self.id, Token.__repr__(self))

    def __str__(self):
        return "{}\t{}".format(self.id, Token.__str__(self))


class MultiWord(Token):
    """[lo, hi | Token.__slots__]"""
    __slots__ = 'lo', 'hi'

    def validate(self, acc=None):
        if acc is None:
            acc = []
        if not is_pos_int(self.lo) \
           or not is_pos_int(self.hi) \
           or self.lo > self.hi:
            acc.append(Fault('id', "{}-{}".format(self.lo, self.hi), "valid"))
        if "_" != self.head:
            acc.append(Fault('head', self.head, "'_'"))
        return Token.validate(self, acc)

    def __init__(self, lo, hi, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lo = lo
        self.hi = hi

    def __eq__(self, other):
        return self is other or isinstance(other, MultiWord) and \
            self.lo == other.lo \
            and self.hi == other.hi \
            and Token.__eq__(self, other)

    def __repr__(self):
        return "MultiWord({}, {}, {})" \
            .format(self.lo, self.hi, Token.__repr__(self))

    def __str__(self):
        return "{}-{}\t{}".format(self.lo, self.hi, Token.__str__(self))


class Sent(object):
    """[(Word | MultiWord)] -> Sent: pos_int -> Word"""
    __slots__ = ('words', 'multi')

    root = Word(0, form="</s>", upostag='ROOT')

    def validate(self, idx):
        acc = []
        assert all(isinstance(w, Word) for w in self.words)
        assert all(isinstance(m, MultiWord) for m in self.multi)
        if not self.words or Sent.root != self.words[0]:
            acc.append(Fault('word', '0', 'root'))
        # check words[1:] and multi-words
        for i, t in enumerate(self):
            old_len = len(acc)
            if old_len < len(t.validate(acc)):
                acc.append("---- in token {}".format(i))
        # more on words
        if not all(i == w.id for i, w in enumerate(self.words)):
            acc.append(Fault('words', 'id', "in order"))
        # more on multi-words
        lo, hi = 0, 0
        for m in self.multi:
            if lo >= m.lo:
                acc.append(
                    Fault('MultiWord', "{}-{}".format(m.lo, m.hi), "sorted"))
            if hi >= m.lo:
                acc.append(
                    Fault('MultiWord', "{}-{}".format(m.lo, m.hi),
                          "non-overlapping"))
            if m.hi > len(self.words):
                acc.append(
                    Fault("MultiWord", "{}-{}".format(m.lo, m.hi),
                          "within bounds"))
            lo, hi = m.lo, m.hi
        if acc:
            acc.append("---------------- in sent {}".format(idx))
        return acc

    def __init__(self, tokens):
        super().__init__()
        words = [Sent.root]
        multi = []
        for t in tokens:
            if isinstance(t, Word):
                words.append(t)
            elif isinstance(t, MultiWord):
                multi.append(t)
            else:
                raise TypeError("expected Word or MultiWord, got {}: {}"
                                .format(type(t), t))
        self.words = tuple(words)
        self.multi = tuple(multi)

    def __eq__(self, other):
        return self is other or isinstance(other, Sent) and \
                len(self.words) == len(other.words) \
                and len(self.multi) == len(other.multi) \
                and all(t == t2 for t, t2 in zip(self, other))

    def __repr__(self):
        return "Sent({})".format(", ".join([repr(t) for t in self]))

    def __str__(self):
        return "\n".join([str(t) for t in chain(self, ("", ""))])

    def iter_words(self):
        """-> iter[Word]; skip root"""
        return islice(self.words, 1, None)

    def __iter__(self):
        w, m = 1, 0
        while w < len(self.words):
            if m < len(self.multi) and w == self.multi[m].lo:
                yield self.multi[m]
                m += 1
            else:
                yield self.words[w]
                w += 1


def load(file):
    """-> iter[Sent]"""
    with open(file, encoding='utf-8') as rdr:
        tokens = []
        for line in rdr:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line:
                tokens.append(Token.parse(line))
            else:
                if tokens:
                    yield Sent(tokens)
                    tokens = []
        if tokens:
            yield Sent(tokens)


def write(sents, file):
    """sents: valid Sent seq"""
    with open(file, 'w', encoding='utf-8') as wtr:
        for s in sents:
            wtr.write(str(s))


def validate(sents):
    """be sents not valid Sent seq, prints explanations"""
    for i, s in enumerate(sents):
        if not isinstance(s, Sent):
            print("----------------", Fault("sent", i, 'Sent'))
            continue
        for r in s.validate(i):
            print(r)


# sents = list(load('/data/ud-treebanks-conll2017/UD_German/de-ud-dev.conllu'))
# write(sents, 'tmp.conllu')
# sents2 = list(load('tmp.conllu'))
# assert sents == sents2

# from glob import glob
# for file in glob("/data/ud-treebanks-conll2017/*/*.conllu"):
#     print("validating", file, "...")
#     validate(load(file))
