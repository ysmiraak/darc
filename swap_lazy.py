from conllu import *
from bisect import insort

class Config:
    """Nivre 2009"""
    __slots__ = ('words','input','stack','graph')

    def __init__(self, sent):
        self.words = sent.words
        self.input = list(range(len(self.words)-1, 0, -1))
        self.stack = [0]
        self.graph = [[] for _ in self.words]

    def is_terminal(self):
        """-> bool"""
        return 0 == len(self.input) and 1 == len(self.stack)

    def at_stack(self, i):
        """Config, int -> int"""
        return self.stack[-(i+1)]

    def at_input(self, i):
        """Config, int -> int"""
        return self.input[-(i+1)]

    def shift(self, _=None):
        """(σ, [i|β], A) ⇒ ([σ|i], β, A)"""
        self.stack.append(self.input.pop())

    def right(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|i], B, A∪{(i, l, j)})"""
        j = self.stack.pop()
        i = self.stack.pop()
        # i -deprel-> j
        w = self.words[j]
        w.head = i
        w.deprel = deprel
        insort(self.graph[i], j)
        self.stack.append(i)

    def left(self, deprel):
        """([σ|i, j], B, A) ⇒ ([σ|j], B, A∪{(j, l, i)})"""
        # only if i != root
        j = self.stack.pop()
        i = self.stack.pop()
        # i <-deprel- j
        w = self.words[i]
        w.head = j
        w.deprel = deprel
        insort(self.graph[j], i)
        self.stack.append(j)

    def swap(self, _=None):
        """([σ|i,j],β,A) ⇒ ([σ|j],[i|β],A)"""
        # only if 0  < i < j
        j = self.stack.pop()
        i = self.stack.pop()
        self.input.append(i)
        self.stack.append(j)

class Oracle:
    """Nivre et al. 2009

    o = Oracle(Sent((Word(1,head=2,form="A"), Word(2,head=3,form="hearing",),
                 Word(3,head=0,form="is"), Word(4,head=3,form="scheduled"),
                 Word(5,head=2,form="on"), Word(6,head=7,form="the"),
                 Word(7,head=5,form="issue"), Word(8,head=4,form="today"),
                 Word(9,head=3,form="."))))

    assert o.order == [0, 1, 2, 6, 7, 3, 4, 5, 8, 9]

    """
    __slots__ = ('words','graph','order')

    def __init__(self, sent):
        self.words = sent.words
        self.graph = [[] for _ in self.words]
        for w in self.words[1:]: self.graph[w.head].append(w.id)
        self.order = [i for i in range(len(self.words))]
        self._order(0, 0)

    def _order(self, i, o):
        past = False
        for j in self.graph[i]:
            if not past and i < j:
                self.order[i] = o
                o += 1
                past = True
            o = self._order(j, o)
        if not past:
            self.order[i] = o
            o += 1
        return o

    def predict(self, config):
        """-> (swap | left | right | shift), (deprel | None)"""
        if 1 == len(config.stack): return Config.shift, None
        j = config.at_stack(0)
        i = config.at_stack(1)
        if self.order[i] > self.order[j]:
            # and not self.input
            # or necessarySwap
            return Config.swap, None
        if i in self.graph[j] and self.graph[i] == config.graph[i]:
            return Config.left, self.words[i].deprel
        if j in self.graph[i] and self.graph[j] == config.graph[j]:
            return Config.right, self.words[j].deprel
        return Config.shift, None

# from copy import deepcopy
# sents = tuple(load('../ud-treebanks-conll2017/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu'))
# s = sents[0]
# sc = deepcopy(s)
# for w in sc.words: w.head = '_'
# o = Oracle(s)
# c = Config(sc)

# while not c.is_terminal():
#     act, arg = o.predict(c)
#     act(c, arg)
