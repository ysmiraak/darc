from itertools import chain
from util import AssocTuple, Fault, is_pos_int, is_nat_int


class AttrVals(AssocTuple):
    """() | iterable -> AttrVals

    assert "_" == str(AttrVals())

    avs = AttrVals((("a","1"),("b","2"),("c","3")))

    s = "a=1|b=2|c=3"

    assert s == str(avs)

    assert avs == AttrVals.parse(s)

    """
    __slots__ = ()

    def __str__(self):
        return "|".join(["{}={}".format(k, v) for k, v in self]) or "_"

    @staticmethod
    def parse(s):
        """-> AttrVals"""
        if "_" == s:
            return AttrVals()
        avs = []
        for p in s.split("|"):
            av = p.split("=")
            if 1 == len(av):
                av = p.split(":")
            if 2 == len(av):
                avs.append(av)
            elif 2 < len(av):
                for a in av[:-1]:
                    avs.append([a, av[-1]])
            elif avs:
                avs[-1][1] += "1" + p
        avs = AttrVals([(a, v) for a, v in avs], True)
        # if s != str(avs): print("invalid avs", s, "parsed as", avs)
        return avs


class Token(object):
    """sum type of Word and MultiWord"""
    __slots__ = ('form', 'lemma', 'upostag', 'xpostag', 'feats', 'head',
                 'deprel', 'deps', 'misc')

    def validate(self):
        """-> [Fault]"""
        res = []
        for a in ('form', 'lemma', 'upostag', 'xpostag', 'deprel', 'deps'):
            v = self.__getattribute__(a)
            if not isinstance(v, str):
                res.append(Fault(a, v, 'str'))
            if "" == v:
                res.append(Fault(a, v, "non-empty"))
        for a in ('feats', 'misc'):
            v = self.__getattribute__(a)
            if not isinstance(v, AttrVals):
                res.append(Fault(a, v, 'AttrVals'))
        return res

    def __init__(self,
                 form="_",
                 lemma="_",
                 upostag="_",
                 xpostag="_",
                 feats=AttrVals(),
                 head="_",
                 deprel="_",
                 deps="_",
                 misc=AttrVals()):
        self.form = form
        self.lemma = lemma
        self.upostag = upostag
        self.xpostag = xpostag
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    def __eq__(self, other):
        return (self is other or isinstance(other, Token) and all(
            self.__getattribute__(a) == other.__getattribute__(a)
            for a in Token.__slots__))

    def __repr__(self):
        return ", ".join(
            [repr(self.__getattribute__(a)) for a in Token.__slots__])

    def __str__(self):
        return "\t".join(
            [str(self.__getattribute__(a)) for a in Token.__slots__])

    @staticmethod
    def parse(s):
        """str -> Word | MultiWord"""
        args = s.split("\t")
        if len(args) != 10:
            raise TypeError("expected 10-col tsv, got: {}".format(s))
        args[5] = AttrVals.parse(args[5])
        args[9] = AttrVals.parse(args[9])
        if "-" in args[0]:
            lo, hi = args[0].split("-")
            args[0] = int(hi)
            tok = MultiWord(int(lo), *args)
        else:
            args[0] = int(args[0])
            args[6] = int(args[6])
            tok = Word(*args)
        return tok


class Word(Token):
    """('id','form','lemma','upostag','xpostag','feats','head','deprel','deps','misc')"""
    __slots__ = 'id'

    def validate(self):
        res = []
        if not is_pos_int(self.id):
            res.append(Fault('id', self.id, "pos-int"))
        if not is_nat_int(self.head):
            res.append(Fault('head', self.head, "nat-int"))
        res.extend(Token.validate(self))
        return res

    def __init__(self, id, *token_args, **token_kwargs):
        self.id = id
        super().__init__(*token_args, **token_kwargs)

    def __eq__(self, other):
        return (self is other or isinstance(other, Word) and
                self.id == other.id and Token.__eq__(self, other))

    # def __lt__(self, other):
    #     return self.id < other.id if isinstance(other, Word) else NotImplemented

    # def __le__(self, other):
    #     return self.id <= other.id if isinstance(other, Word) else NotImplemented

    # def __gt__(self, other):
    #     return self.id > other.id if isinstance(other, Word) else NotImplemented

    # def __ge__(self, other):
    #     return self.id >= other.id if isinstance(other, Word) else NotImplemented

    def __repr__(self):
        return "Word({}, {})".format(self.id, Token.__repr__(self))

    def __str__(self):
        return "{}\t{}".format(self.id, Token.__str__(self))


class MultiWord(Token):
    """('lo','hi','form','lemma','upostag','xpostag','feats','head','deprel','deps','misc')"""
    __slots__ = ('lo', 'hi')

    def validate(self):
        res = []
        if not is_pos_int(self.lo) or not is_pos_int(
                self.hi) or self.lo > self.hi:
            res.append(
                Fault('id', "{}-{}".format(self.lo, self.hi), "invalid"))
        if "_" != self.head:
            res.append(Fault('head', self.head, "'_'"))
        res.extend(Token.validate(self))
        return res

    def __init__(self, lo, hi, *token_args, **token_kwargs):
        self.lo = lo
        self.hi = hi
        super().__init__(*token_args, **token_kwargs)

    def __eq__(self, other):
        return (self is other or isinstance(other, MultiWord) and
                self.lo == other.lo and self.hi == other.hi and
                Token.__eq__(self, other))

    def __repr__(self):
        return "MultiWord({}, {}, {})".format(self.lo, self.hi,
                                              Token.__repr__(self))

    def __str__(self):
        return "{}-{}\t{}".format(self.lo, self.hi, Token.__str__(self))


class Sent(object):
    """[(Word | MultiWord)] -> Sent: pos_int -> Word"""
    __slots__ = ('words', 'multi')

    root = Word(0, form="</s>", upostag='ROOT')

    def validate(self):
        res = []
        # check words[1:] and multi-words
        for tok in self:
            faults = tok.validate()
            if faults:
                res.extend(faults)
                res.append("---- in " + repr(tok))
        # more on words
        if not all(i == w.id for i, w in enumerate(self.words)):
            res.append(Fault('Word', "ids", "in order"))
        # more on multi-words
        lo, hi = 0, 0
        for m in self.multi:
            if lo >= m.lo:
                res.append(Fault('MultiWord', "spans", "sorted"))
            if hi >= m.lo:
                res.append(Fault('MultiWord', "spans", "non-overlapping"))
            if m.hi > len(self.words):
                res.append(
                    Fault("MultiWord span", "{}-{}".format(m.lo, m.hi),
                          "within bounds"))
            lo, hi = m.lo, m.hi
        return res

    def __init__(self, tokens):
        words = [Sent.root]
        multi = []
        for t in tokens:
            if isinstance(t, Word):
                words.append(t)
            elif isinstance(t, MultiWord):
                multi.append(t)
            else:
                raise TypeError("expected Word or MultiWord, got {}: {}".
                                format(type(t), t))
        self.words = tuple(words)
        self.multi = tuple(multi)

    def __eq__(self, other):
        return (self is other or isinstance(other, Sent) and
                len(self.words) == len(other.words) and
                len(self.multi) == len(other.multi) and
                all(t == t2 for t, t2 in zip(self, other)))

    def __repr__(self):
        return "Sent({})".format(", ".join([repr(t) for t in self]))

    def __str__(self):
        return "\n".join([str(t) for t in chain(self, ("", ""))])

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
    """-> lazy[Sent]"""
    with open(file, encoding='utf-8') as rdr:
        tokens = []
        for line in rdr:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line:
                t = Token.parse(line)
                tokens.append(t)
            else:
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
        res = s.validate()
        for r in res:
            print(r)
        if res:
            print("---------------- in sent", i)


# sents = tuple(load('/data/ud-treebanks-conll2017/UD_German/de-ud-dev.conllu'))
# write(sents, 'tmp.conllu')
# sents2 = tuple(load('tmp.conllu'))
# assert sents == sents2

# from glob import glob
# for file in glob("/data/ud-treebanks-conll2017/*/*.conllu"):
#     print("validating", file, "...")
#     validate(load(file))
