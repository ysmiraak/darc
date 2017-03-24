from itertools import repeat, islice
from bisect import bisect_left, insort_right


class Config(object):
    """active nodes: i,j = config.stack[-2:]"""
    __slots__ = 'words', 'input', 'stack', 'graph'

    def __init__(self, sent):
        super().__init__()
        n = len(sent.words)
        self.words = sent.words
        self.input = list(range(n - 1, 0, -1))
        self.stack = [0]
        self.graph = [[] for _ in range(n)]

    terminal = [0], []

    def is_terminal(self):
        """-> bool"""
        return Config.terminal == (self.stack, self.input)

    def doable(self, act):
        """-> bool; act: 'shift' | 'right' | 'left' | 'swap'"""
        if 'shift' == act:
            return 0 != len(self.input)
        elif 'right' == act:
            return 2 <= len(self.stack)
        # and (0 != self.stack[-2] or not self.graph[0])
        elif 'left' == act:
            return 2 <= len(self.stack) and 0 != self.stack[-2]
        elif 'swap' == act:
            return 2 <= len(self.stack) and 0 < self.stack[-2] < self.stack[-1]
        else:
            raise TypeError("unknown act: {}".format(act))

    def shift(self, _=None, __=True):
        """(σ, [i|β], A) ⇒ ([σ|i], β, A)"""
        self.stack.append(self.input.pop())

    def right(self, label=None, mutate=True):
        """([σ|i, j], B, A) ⇒ ([σ|i], B, A∪{(i, l, j)})"""
        j = self.stack.pop()
        i = self.stack[-1]
        insort_right(self.graph[i], j)
        # i -deprel-> j
        if mutate:
            self.words[j].head = i
            if label:
                self.words[j].deprel = label

    def left(self, label=None, mutate=True):
        """([σ|i, j], B, A) ⇒ ([σ|j], B, A∪{(j, l, i)})"""
        j = self.stack[-1]
        i = self.stack.pop(-2)
        insort_right(self.graph[j], i)
        # i <-deprel- j
        if mutate:
            self.words[i].head = j
            if label:
                self.words[i].deprel = label

    def swap(self, _=None, __=True):
        """([σ|i,j],β,A) ⇒ ([σ|j],[i|β],A)"""
        self.input.append(self.stack.pop(-2))


class Oracle(object):
    """six possible modes:

    0. projective=True, arc-standard

    1. lazy=False, non-projective with swap (Nivre 2009)

    2. default, lazy swap (Nivre, Kuhlmann, Hall 2009)

    plus three unlabeled variants

    """
    __slots__ = 'mode6', 'graph', 'order', 'mpcrt', 'drels'

    def __init__(self, gold, projective=False, lazy=True, labeled=True):
        super().__init__()
        self.mode6 = 0
        n = len(gold.words)
        self.graph = [[] for _ in range(n)]
        for w in islice(gold.words, 1, None):
            self.graph[w.head].append(w.id)
        if projective:
            return
        self.mode6 = 1
        self.order = list(range(n))
        Oracle._order(self, 0, 0)
        if not lazy:
            return
        self.mode6 = 3
        self.mpcrt = list(repeat(-1, n))
        config = Config(gold)
        while not config.is_terminal():
            act, arg = Oracle.predict(self, config)
            if 'shift' == act and not config.input:
                break
            getattr(config, act)(arg, False)
        Oracle._mpcrt(self, config.graph, 0, 0)
        self.mode6 = 2
        if labeled:
            self.drels = [w.deprel for w in gold.words]
        else:
            self.mode6 += 3

    def _order(self, n, o):
        # in-order traversal ordering
        i = bisect_left(self.graph[n], n)
        for c in islice(self.graph[n], i):
            o = Oracle._order(self, c, o)
        self.order[n] = o
        o += 1
        for c in islice(self.graph[n], i, None):
            o = Oracle._order(self, c, o)
        return o

    def _mpcrt(self, g, n, r):
        # maximal projective component root
        self.mpcrt[n] = r
        i = 0
        for c in self.graph[n]:
            if -1 != self.mpcrt[c]:
                continue
            i = bisect_left(g[n], c, i)
            Oracle._mpcrt(self, g, c, r
                          if i < len(g[n]) and c == g[n][i] else c)

    def predict(self, config):
        """-> act: str, arg: (str | None)

        act: 'shift' | 'right' | 'left' (| 'swap')

        getattr(config, act)(arg)

        """
        mode3 = self.mode6 % 3
        if 1 == len(config.stack):
            return 'shift', None
        j = config.stack[-1]
        i = config.stack[-2]
        if 0 != mode3 and self.order[i] > self.order[j]:
            if 1 == mode3:
                return 'swap', None
            if not config.input \
               or self.mpcrt[j] != self.mpcrt[config.input[-1]]:
                return 'swap', None
        if i in self.graph[j] and self.graph[i] == config.graph[i]:
            return 'left', self.drels[i] if 3 > self.mode6 else None
        if j in self.graph[i] and self.graph[j] == config.graph[j]:
            return 'right', self.drels[j] if 3 > self.mode6 else None
        return 'shift', None


# from conllu import Word, Sent

# s = Sent((Word(1, head=2, deprel='DET', form="A"), Word(
#     2,
#     head=3,
#     deprel='SBJ',
#     form="hearing", ), Word(3, head=0, deprel='ROOT', form="is"),
#           Word(4, head=3, deprel='VG', form="scheduled"),
#           Word(5, head=2, deprel='NMOD', form="on"),
#           Word(6, head=7, deprel='DET', form="the"),
#           Word(7, head=5, deprel='PC', form="issue"),
#           Word(8, head=4, deprel='ADV', form="today"),
#           Word(9, head=3, deprel='P', form=".")))
# o = Oracle(s)
# assert o.order == [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]
# assert o.mpcrt == [0, 2, 2, 3, 4, 5, 5, 5, 8, 9]

# s = Sent((Word(1,head=7,deprel='NMOD',form="Who"),
#           Word(2,head=0,deprel='ROOT',form="did",),
#           Word(3,head=2,deprel='SUBJ',form="you"),
#           Word(4,head=2,deprel='VG',form="send"),
#           Word(5,head=6,deprel='DET',form="the"),
#           Word(6,head=4,deprel='OBJ1',form="letter"),
#           Word(7,head=4,deprel='OBJ2',form="to"),
#           Word(8,head=2,deprel='P',form="?")))
# o = Oracle(s)
# assert o.order == [0, 6, 1, 2, 3, 4, 5, 7, 8]
# assert o.mpcrt == [0, 1, 2, 2, 4, 4, 4, 7, 8]

# from copy import deepcopy
# def test_oracle(s, verbose=True, proj=False, lazy=True):
#     sc = deepcopy(s)
#     for w in sc.words:
#         w.head = '_'
#     assert s != sc
#     o = Oracle(s, proj, lazy)
#     c = Config(sc)
#     while not c.is_terminal():
#         act, arg = o.predict(c)
#         if verbose:
#             print(act, arg)
#         getattr(c, act)(arg)
#     assert s == sc

# from conllu import load
# from glob import glob
# for file in glob("/data/ud-treebanks-conll2017/*/*.conllu"):
#     print("testing oracle on", file, "...")
#     for i,s in enumerate(load(file)):
#         try:
#             test_oracle(s, False)
#         except Exception:
#             print("failed at sent", i)
