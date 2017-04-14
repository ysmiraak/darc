from sys import exit
from conllu import load, save

def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="the graph is darc and full of errors.")
    parser.add_argument('--model', help="npy model file to save with --train and to load with --parse")
    parser.add_argument('--parse', help="conllu files to be parsed", nargs='*')
    parser.add_argument('--outfile', help="paths for the parsed files", nargs='*')
    parser.add_argument('--concat-outfiles', action='store_true')
    parser.add_argument('--train', help="conllu files for training", nargs='*')
    parser.add_argument('--param', help="json parameter file for training")
    parser.add_argument('--form-w2v', help="word2vec file for form embeddings")
    parser.add_argument('--form-w2v-is-binary', action='store_true')
    parser.add_argument('--lemm-w2v', help="word2vec file for lemma embeddings")
    parser.add_argument('--lemm-w2v-is-binary', action='store_true')
    parser.add_argument('--verbose', type=int, help="0, 1, or 2", default=1)
    args = parser.parse_args()
    if args.train:
        if not args.param:
            exit("cannot train without a param file")
        if not args.model:
            if args.parse:
                if args.verbose:
                    print("will train without saving a model")
            else:
                exit("refuse to train in vain")
        if not args.form_w2v:
            exit("cannot train without form embeddings")
        if not args.lemm_w2v:
            if args.verbose:
                print("will train without lemma embeddings")
    elif args.parse:
        if not args.outfile:
            exit("no where to save the parsed files")
        elif args.concat_outfiles:
            if 1 < len(args.outfile):
                exit("cannot concat into multiple outfiles")
        elif len(args.parse) != len(args.outfile):
            exit("files for --parse and --outfile don't match, consider --concat-outfiles")
        if args.param:
            exit("param supplied but no training to do")
        if args.parse and not args.model:
            exit("cannot parse without a model")
        if args.form_w2v or args.lemm_w2v:
            exit("embeddings supplied but no training to do")
    else:
        exit("nothing to be done")
    return args


def parse_params(file):
    """-> bool: proj, epochs: int, params: dict"""
    import json
    with open(file, encoding='utf-8') as file:
        params = json.load(file)
    if 'proj' in params:
        proj = params['proj']
        del params['proj']
    else:
        exit("proj uspecified in param file")
    if 'epochs' in params:
        epochs = params['epochs']
        del params['epochs']
    else:
        exit("epochs uspecified in param file")
    return proj, epochs, params


def setup_model(args):
    """-> Setup | SetupNoLemma, keras.models.Model"""
    import numpy as np
    from setup import Setup
    from setup_no_lemma import SetupNoLemma
    if args.train:
        from gensim.models.keyedvectors import KeyedVectors
        proj, epochs, params = parse_params(args.param)
        sents = [sent for train in args.train for sent in load(train)]
        form_w2v = KeyedVectors.load_word2vec_format(args.form_w2v, binary=args.form_w2v_is_binary)
        if args.lemm_w2v:
            lemm_w2v = KeyedVectors.load_word2vec_lemmat(args.lemm_w2v, binary=args.lemm_w2v_is_binary)
            setup = Setup.cons(sents=sents, proj=proj, form_w2v=form_w2v, lemm_w2v=lemm_w2v)
        else:
            setup = Setup.cons(sents=sents, proj=proj, form_w2v=form_w2v)
        model = setup.model(**params)
        setup.train(model, verbose=args.verbose, epochs=epochs)
        np.save(args.model,
                {'setup': setup.bean(with_data=False),
                 'model': model.to_json(),
                 'weights': model.get_weights()})
    else:
        from keras.models import model_from_json
        bean = np.load(args.model).item()
        setup = (Setup if 'lemm2idx' in bean['setup'] else SetupNoLemma)(bean['setup'])
        model = model_from_json(bean['model'])
        model.set_weights(bean['weights'])
    return setup, model


if '__main__' == __name__:
    args = parse_args()
    setup, model = setup_model(args)
    if args.parse:
        if args.concat_outfiles:
            save((setup.parse(sent, model) for file in args.parse for sent in load(file)), args.outfile[0])
        else:
            for infile, outfile in zip(args.parse, args.outfile):
                save((setup.parse(sent, model) for sent in load(infile)), outfile)
