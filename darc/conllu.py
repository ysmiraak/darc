from collections import namedtuple


cols = 'id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'
Sent = namedtuple('Sent', cols + ('multi', ))
Sent.cols = cols
del cols
Sent.obsc = "_"
Sent.root = "\xa0"
Sent.dumb = ""


def cons(lines, dumb=Sent.dumb, udrel=True):
    """[str] -> Sent"""
    multi = []
    nodes = [[0, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb, dumb]]
    for line in lines:
        node = line.split("\t")
        assert 10 == len(node)
        try:
            node[0] = int(node[0])
        except ValueError:
            if "-" in node[0]:
                multi.append(line)
        else:
            try:  # head might be empty for interim results
                node[6] = int(node[6])
            except ValueError:
                pass
            if udrel:
                try:  # acl:relcl -> acl
                    node[7] = node[7][:node[7].index(":")]
                except ValueError:
                    pass
            nodes.append(node)
    return Sent(*zip(*nodes), tuple(multi))


Sent.cons = cons
del cons


def load(file, dumb=Sent.dumb, udrel=True):
    """-> iter([Sent])"""
    with open(file, encoding='utf-8') as file:
        sent = []
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                pass
            elif line:
                sent.append(line.replace(" ", "\xa0"))
            elif sent:
                yield Sent.cons(sent, dumb, udrel)
                sent = []
        if sent:
            yield Sent.cons(sent, dumb, udrel)


def save(sents, file):
    """sents: [Sent]"""
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            multi_idx = [int(multi[:multi.index("-")]) for multi in sent.multi]
            w, m = 1, 0
            while w < len(sent.id):
                if m < len(multi_idx) and w == multi_idx[m]:
                    line = sent.multi[m]
                    m += 1
                else:
                    line = "\t".join([str(getattr(sent, col)[w]) for col in Sent.cols])
                    w += 1
                file.write(line.replace("\xa0", " "))
                file.write("\n")
            file.write("\n")


# from darc import ud2
# sents = list(load(ud2.path('de', 'dev')))
# save(sents, "../lab/tmp.conllu")
# sents2 = list(load("../lab/tmp.conllu"))
# assert sents == sents2
