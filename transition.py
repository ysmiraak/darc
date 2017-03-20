from itertools import repeat, islice
from bisect import bisect_left, insort_right


class Config(object):
    """active nodes: i,j = config.stack[-2:]"""
    __slots__ = ('words', 'input', 'stack', 'graph')

    def __init__(self, sent):
        n = len(sent.words)
        self.words = sent.words
        self.input = list(range(n - 1, 0, -1))
        self.stack = [0]
        self.graph = [[] for _ in range(n)]

    def is_terminal(self):
        """-> bool"""
        return 0 == len(self.input) and 1 == len(self.stack)

    def shift(self, _=None):
        """(σ, [i|β], A) ⇒ ([σ|i], β, A)"""
        # if input
        self.stack.append(self.input.pop())

    def right(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|i], B, A∪{(i, l, j)})"""
        # if 2 <= len(stack)
        j = self.stack.pop()
        i = self.stack[-1]
        # i -deprel-> j
        w = self.words[j]
        w.head = i
        w.deprel = deprel
        insort_right(self.graph[i], j)

    def left(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|j], B, A∪{(j, l, i)})"""
        # if 2 <= len(stack) and i != 0
        j = self.stack[-1]
        i = self.stack.pop(-2)
        # i <-deprel- j
        w = self.words[i]
        w.head = j
        w.deprel = deprel
        insort_right(self.graph[j], i)

    def swap(self, _=None):
        """([σ|i,j],β,A) ⇒ ([σ|j],[i|β],A)"""
        # if 0 != i and i < j (< 0 i j)
        self.input.append(self.stack.pop(-2))


class Oracle(object):
    """three possible modes:

    0. proj=True, arc-standard projective

    1. lazy=False, non-projective with swap (Nivre 2009)

    2. default, lazy swap (Nivre, Kuhlmann, Hall 2009)

    """
    __slots__ = ('words', 'graph', 'mode3', 'order', 'mpcrt')

    def __init__(self, gold, proj=False, lazy=True):
        self.mode3 = 0
        n = len(gold.words)
        self.words = gold.words
        self.graph = [[] for _ in range(n)]
        for w in self.words[1:]:
            self.graph[w.head].append(w.id)
        if proj:
            return
        self.mode3 = 1
        self.order = list(repeat(0, n))
        self._order(0, 0)
        if not lazy:
            return
        self.mode3 = 0
        self.mpcrt = list(repeat(-1, n))
        config = Config(gold)
        while not config.is_terminal():
            act, arg = self.predict(config)
            if act == Config.shift and not config.input:
                break
            act(config, arg)
        self._mpcrt(config.graph, 0, 0)
        self.mode3 = 2

    def _order(self, n, o):
        # in-order traversal ordering
        i = bisect_left(self.graph[n], n)
        for c in islice(self.graph[n], i):
            o = self._order(c, o)
        self.order[n] = o
        o += 1
        for c in islice(self.graph[n], i, None):
            o = self._order(c, o)
        return o

    def _mpcrt(self, g, n, r):
        # maximal projective component root
        self.mpcrt[n] = r
        i = 0
        for c in self.graph[n]:
            if -1 != self.mpcrt[c]:
                continue
            i = bisect_left(g[n], c, i)
            self._mpcrt(g, c, r if i < len(g[n]) and c == g[n][i] else c)

    def predict(self, config):
        """-> (shift | right | left (| swap)), (deprel | None)"""
        if 1 == len(config.stack):
            return Config.shift, None
        j = config.stack[-1]
        i = config.stack[-2]
        if self.mode3 != 0 and self.order[i] > self.order[j]:
            if self.mode3 == 1:
                return Config.swap, None
            if (not config.input or
                    self.mpcrt[j] != self.mpcrt[config.input[-1]]):
                return Config.swap, None
        if self.words[i].head == j and self.graph[i] == config.graph[i]:
            return Config.left, self.words[i].deprel
        if i == self.words[j].head and self.graph[j] == config.graph[j]:
            return Config.right, self.words[j].deprel
        return Config.shift, None


# from conllu import Sent, load

# s = Sent((Word(1,head=2,deprel='DET',form="A"),
#           Word(2,head=3,deprel='SBJ',form="hearing",),
#           Word(3,head=0,deprel='ROOT',form="is"),
#           Word(4,head=3,deprel='VG',form="scheduled"),
#           Word(5,head=2,deprel='NMOD',form="on"),
#           Word(6,head=7,deprel='DET',form="the"),
#           Word(7,head=5,deprel='PC',form="issue"),
#           Word(8,head=4,deprel='ADV',form="today"),
#           Word(9,head=3,deprel='P',form=".")))
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
#     for w in sc.words: w.head = '_'
#     assert s != sc
#     o = Oracle(s, proj, lazy)
#     c = Config(sc)
#     while not c.is_terminal():
#         act,arg = o.predict(c)
#         if verbose: print(act.__name__, arg)
#         act(c,arg)
#     assert s == sc

# from glob import glob
# for file in glob("/data/ud-treebanks-conll2017/*/*.conllu"):
#     print("testing oracle on", file, "...")
#     for i,s in enumerate(load(file)):
#         try:
#             test_oracle(s, False)
#         except Exception:
#             print("failed at sent", i)
