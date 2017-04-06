from collections import namedtuple


cols = 'id', 'form', 'lemma', 'upostag', 'xpostag', \
       'feats', 'head', 'deprel', 'deps', 'misc'

Sent = namedtuple('Sent', ('multi', ) + cols)

Sent.cols = cols

del cols

Sent.dumb = 0, "", "", "", "", "", 0, "", "", ""


def sent(lines):
    """[str] -> Sent"""
    multi = []
    cols = [[x] for x in Sent.dumb]
    for line in lines:
        args = line.split("\t")
        assert 10 == len(args)
        if "-" in args[0]:
            multi.append(args)
        else:
            args[0] = int(args[0])
            args[7] = args[7].split(":")[0]  # acl:relcl -> acl
            if "_" != args[6]:  # head might be empty for interim results
                args[6] = int(args[6])
            for col, val in zip(cols, args):
                col.append(val)
    return Sent(multi, *cols)


def load(file):
    """-> iter([Sent])"""
    with open(file, encoding='utf-8') as file:
        lines = []
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                pass
            elif line:
                lines.append(line)
            elif lines:
                yield sent(lines)
                lines = []
        if lines:
            yield sent(lines)


def write(sents, file):
    """sents: [Sent]"""
    with open(file, 'w', encoding='utf-8') as file:
        for sent in sents:
            multi_idx = [int(multi[0].split("-")[0]) for multi in sent.multi]
            w, m = 1, 0
            while w < len(sent.id):
                if m < len(multi_idx) and w == multi_idx[m]:
                    file.write("\t".join([str(x) for x in sent.multi[m]]))
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
