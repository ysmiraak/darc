from collections import namedtuple


cols = 'id', 'form', 'lemma', 'upostag', 'xpostag', \
       'feats', 'head', 'deprel', 'deps', 'misc'
Sent = namedtuple('Sent', ('multi', ) + cols)
Sent.cols = cols
del cols

# id=0 serves as root; form="", upostag="", feats="" used by setup as sentinel
# for missing nodes; head=0, deprel="_" used by transition as default for
# consistency; the others serve no purpose.
Sent.dumb = 0, "", None, "", None, "", 0, "_", None, None


def build(lines):
    """[str] -> Sent"""
    multi = []
    nodes = [Sent.dumb]
    for line in lines:
        node = line.split("\t")
        assert 10 == len(node)
        try:
            node[0] = int(node[0])
        except ValueError:
            if "-" in node[0]:
                multi.append(line)
                continue
        try:  # head might be empty for interim results
            node[6] = int(node[6])
        except ValueError:
            pass
        try:  # acl:relcl -> acl
            node[7] = node[7][:node[7].index(":")]
        except ValueError:
            pass
        nodes.append(node)
    return Sent(tuple(multi), *zip(*nodes))


Sent.build = build
del build


def load(file):
    """-> iter([Sent])"""
    with open(file, encoding='utf-8') as file:
        sent = []
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                pass
            elif line:
                sent.append(line)
            elif sent:
                yield Sent.build(sent)
                sent = []
        if sent:
            yield Sent.build(sent)


def write(sents, file):
    """sents: [Sent]"""
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            multi_idx = [int(multi[:multi.index("-")]) for multi in sent.multi]
            w, m = 1, 0
            while w < len(sent.id):
                if m < len(multi_idx) and w == multi_idx[m]:
                    file.write(sent.multi[m])
                    file.write("\n")
                    m += 1
                else:
                    file.write("\t".join(
                        [str(getattr(sent, col)[w]) for col in Sent.cols]))
                    file.write("\n")
                    w += 1
            file.write("\n")


# sents = list(load("/data/ud-treebanks-conll2017/UD_German/de-ud-dev.conllu"))
# write(sents, ".tmp/tmp.conllu")
# sents2 = list(load(".tmp/tmp.conllu"))
# assert sents == sents2
