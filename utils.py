def is_pos_int(x):
    """-> bool"""
    return isinstance(x, int) and 0 < x

def is_nat_int(x):
    """-> bool"""
    return isinstance(x, int) and 0 <= x

class Fault(object):
    """like Exception but for collecting and not throwing"""
    __slots__ = ('attr','val','spec')

    def __init__(self, attr, val, spec):
        self.attr = attr
        self.val = val
        self.spec = spec

    def __repr__(self):
        return "Fault({}, {}, {})".format(self.attr, self.val, self.spec)

    def __str__(self):
        return "{} {} be not {}".format(self.attr, self.val, self.spec)

class AssocTuple(tuple):
    """() | iterable -> AssocTuple; useful as small immutable dict"""
    __slots__ = ()

    @staticmethod
    def __new__(cls, iterable=(), unsafe=False):
        if unsafe: return super(AssocTuple, cls).__new__(cls, iterable)
        if not iterable:
            if not hasattr(cls, 'nil'):
                cls.nil = cls(unsafe=True)
            return cls.nil
        kvs = []
        ks = set()
        for kv in iterable:
            if not hasattr(kv, '__len__') or 2 != len(kv):
                raise TypeError("AssocTuple expected pairs, got: {}".format(iterable))
            elif kv[0] in ks:
                raise TypeError("duplicate keys for AssocTuple: {}".format(iterable))
            else:
                ks.add(kv[0])
                kvs.append(tuple(kv))
        return super(AssocTuple, cls).__new__(cls, kvs)

    def sort(self, by=None, rev=False):
        """-> AssocTuple"""
        return type(self)(sorted(self, key=by, reverse=rev), True)

    def index(self, k):
        i = 0
        for q,_ in self:
            if q == k: return i
            i += 1
        return None

    def dissoc(self, k):
        """-> AssocTuple; O(len)"""
        i = AssocTuple.index(self, k)
        if None == i: return self
        t = tuple(self)
        return type(self)(t[:i] + t[i+1:], True)

    def assoc(self, k, v):
        """-> AssocTuple; O(len)"""
        t = tuple(self)
        i = AssocTuple.index(self, k)
        if None == i: return type(self)(t + ((k,v),), True)
        return type(self)(t[:i] + ((k,v),) + t[i+1:], True)

    def __add__(self, other):
        for q,_ in other:
            self = self.dissoc(q)
        return AssocTuple(tuple.__add__(self, other), True)

    def __contains__(self, k):
        return None != AssocTuple.index(self, k)

    def __getitem__(self, k):
        i = AssocTuple.index(self, k)
        if None == i: raise KeyError(k)
        return tuple.__getitem__(self, i)[1]
