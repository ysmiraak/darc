def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="produce training data for word2vec.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -v")
    parser.add_argument('--data', required=True, nargs='+', help="conllu files to use")
    parser.add_argument('--form', nargs='+', help="files to save the forms")
    parser.add_argument('--lemm', nargs='+', help="files to save the lemmas")
    args = parser.parse_args()
    from sys import exit
    if args.form:
        if len(args.data) != len(args.form):
            exit("the number of data files does not match the number of form files")
    elif args.lemm:
        if len(args.data) != len(args.form):
            exit("the number of data files does not match the number of lemm files")
    else:
        exit("nothing to be done")
    return args


def extract(conllu, form, lemm, verbose, root, load, counter):
    if verbose:
        print("loading", conllu, "....")
    sents = list(load(conllu, dumb=root))
    if form:
        freq = counter(form for sent in sents for form in sent.form)
        with open(form, 'w', encoding='utf-8') as file:
            for sent in sents:
                file.write(" ".join([form if 1 != freq[form] else "_" for form in sent.form]))
                file.write("\n")
        if verbose:
            print("written", form, "....")
        del freq
    if lemm:
        freq = counter(lemm for sent in sents for lemm in sent.lemma)
        with open(lemm, 'w', encoding='utf-8') as file:
            for sent in sents:
                file.write(" ".join([lemm if 1 != freq[lemm] else "_" for lemm in sent.lemma]))
                file.write("\n")
        if verbose:
            print("written", lemm, "....")


if '__main__' == __name__:
    args = parse_args()
    from itertools import repeat
    if not args.form:
        args.form = repeat(None)
    if not args.lemm:
        args.lemm = repeat(None)
    from conllu import Sent, load
    from collections import Counter
    for conllu, form, lemm in zip(args.data, args.form, args.lemm):
        extract(conllu, form, lemm, verbose, Sent.root, load, Counter)
