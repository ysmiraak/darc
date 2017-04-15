def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="train a darc parser.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -vv")
    parser.add_argument('--model', required=True, help="npy model file to save")
    parser.add_argument('--train', required=True, nargs='+', help="conllu files for training")
    parser.add_argument('--form-w2v', required=True, help="word2vec file for form embeddings")
    parser.add_argument('--form-w2v-is-binary', action='store_true')
    parser.add_argument('--lemm-w2v', help="word2vec file for lemma embeddings")
    parser.add_argument('--lemm-w2v-is-binary', action='store_true')
    parser.add_argument('--proj', action='store_true', help="train a projective parser")
    parser.add_argument('--upos-embed-dim', type=int, default=12, help="default: 12")
    parser.add_argument('--drel-embed-dim', type=int, default=16, help="default: 16")
    parser.add_argument('--hidden-units', type=int, default=256, help="default: 256")
    parser.add_argument('--embed-init', default='uniform', help="default: uniform")
    parser.add_argument('--dense-init', default='orthogonal', help="default: orthogonal")
    parser.add_argument('--embed-const', default='unitnorm', help="default: unitnorm")
    parser.add_argument('--dense-const', default=None, help="default: None")
    parser.add_argument('--embed-dropout', type=float, default=0.25, help="default: 0.25")
    parser.add_argument('--dense-dropout', type=float, default=0.25, help="default: 0.25")
    parser.add_argument('--activation', default='relu', help="default: relu")
    parser.add_argument('--optimizer', default='adamax', help="default: adamax")
    parser.add_argument('--epochs', type=int, default=12, help="default: 12")
    args = parser.parse_args()
    if not args.verbose:
        args.verbose = 0
    elif 1 == args.verbose:
        args.verbose = 2
    elif 2 <= args.verbose:
        args.verbose = 1
    return args


def make_setup(train, form_w2v, form_w2v_is_binary, lemm_w2v, lemm_w2v_is_binary, proj, verbose):
    """-> Setup | SetupNoLemma"""
    if verbose:
        print("training a", "projective" if proj else "non-projective", "parser")
        print("loading", *train, "....")
    from conllu import load
    sents = [sent for train in train for sent in load(train)]
    if verbose:
        print("loading", form_w2v, "....")
    from gensim.models.keyedvectors import KeyedVectors
    form_w2v = KeyedVectors.load_word2vec_format(form_w2v, binary=form_w2v_is_binary)
    if lemm_w2v:
        if verbose:
            print("loading", lemm_w2v, "....")
        lemm_w2v = KeyedVectors.load_word2vec_lemmat(lemm_w2v, binary=lemm_w2v_is_binary)
        if verbose:
            print("preparing training data ....")
        from setup import Setup
        setup = Setup.cons(sents=sents, proj=proj, form_w2v=form_w2v, lemm_w2v=lemm_w2v)
    else:
        if verbose:
            print("preparing training data ....")
        from setup_no_lemma import SetupNoLemma
        setup = Setup.cons(sents=sents, proj=proj, form_w2v=form_w2v)
    return setup


if '__main__' == __name__:
    args = parse_args()
    setup = make_setup(
        train=args.train,
        form_w2v=args.form_w2v,
        form_w2v_is_binary=args.form_w2v_is_binary,
        lemm_w2v=args.lemm_w2v,
        lemm_w2v_is_binary=args.lemm_w2v_is_binary,
        proj=args.proj,
        verbose=args.verbose)
    model = setup.model(
        upos_embed_dim=args.upos_embed_dim,
        drel_embed_dim=args.drel_embed_dim,
        hidden_units=args.hidden_units,
        embed_init=args.embed_init,
        dense_init=args.dense_init,
        embed_const=args.embed_const,
        dense_const=args.dense_const,
        embed_dropout=args.embed_dropout,
        dense_dropout=args.dense_dropout,
        activation=args.activation,
        optimizer=args.optimizer)
    setup.train(model, epochs=args.epochs, verbose=args.verbose)
    import numpy as np
    np.save(args.model,
            {'setup': setup.bean(with_data=False),
             'model': model.to_json(),
             'weights': model.get_weights()})
    if args.verbose:
        print("saved model", args.model)


# TODO: adjust to refactoring
