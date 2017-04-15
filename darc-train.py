def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="train a darc parser.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -vv")
    parser.add_argument('--model', required=True, help="npy model file to save")
    parser.add_argument('--train', required=True, nargs='+', help="conllu files for training")
    parser.add_argument('--form-w2v', required=True, help="word2vec file for form embeddings")
    parser.add_argument('--lemm-w2v', help="word2vec file for lemma embeddings")
    parser.add_argument('--w2v-is-binary', action='store_true')
    parser.add_argument('--proj', action='store_true', help="train a projective parser")
    parser.add_argument('--upos-embed-dim', type=int, default=12, help="default: 12")
    parser.add_argument('--drel-embed-dim', type=int, default=16, help="default: 16")
    parser.add_argument('--embed-init', default='uniform', help="default: uniform")
    parser.add_argument('--embed-const', default='unitnorm', help="default: unitnorm")
    parser.add_argument('--embed-dropout', type=float, default=0.25, help="default: 0.25")
    parser.add_argument('--hidden-layers', type=int, default=2, help="default: 2")
    parser.add_argument('--hidden-units', type=int, default=256, help="default: 256")
    parser.add_argument('--hidden-bias', type=float, default=0.0, help="default: 0.0")
    parser.add_argument('--hidden-init', default='orthogonal', help="default: orthogonal")
    parser.add_argument('--hidden-const', default='none', help="default: none")
    parser.add_argument('--hidden-dropout', type=float, default=0.25, help="default: 0.25")
    parser.add_argument('--activation', default='relu', help="default: relu")
    parser.add_argument('--output-init', default='orthogonal', help="default: orthogonal")
    parser.add_argument('--output-const', default='none', help="default: none")
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


def make_setup(train, proj, form_w2v, lemm_w2v, w2v_is_binary, verbose):
    """-> Setup"""
    if verbose:
        print("training a", "projective" if proj else "non-projective", "parser")
        print("loading", *train, "....")
    from darc import conllu
    sents = [sent for train in train for sent in conllu.load(train)]
    if verbose:
        print("loading", form_w2v, "....")
    from gensim.models.keyedvectors import KeyedVectors
    form_w2v = KeyedVectors.load_word2vec_format(form_w2v, binary=w2v_is_binary)
    if lemm_w2v:
        if verbose:
            print("loading", lemm_w2v, "....")
        lemm_w2v = KeyedVectors.load_word2vec_lemmat(lemm_w2v, binary=w2v_is_binary)
    else:
        lemm_w2v = None
    if verbose:
        print("preparing training data ....")
    from darc.setup import Setup
    return Setup.cons(sents=sents, proj=proj, form_w2v=form_w2v, lemm_w2v=lemm_w2v)


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
        embed_init=args.embed_init,
        embed_const=args.embed_const,
        embed_dropout=args.embed_dropout,
        hidden_layers=args.hidden_layers,
        hidden_units=args.hidden_units,
        hidden_bias=args.hidden_bias,
        hidden_init=args.hidden_init,
        hidden_const=args.hidden_const,
        hidden_dropout=args.hidden_dropout,
        activation=args.activation,
        output_init=args.output_init,
        output_const=args.output_const,
        optimizer=args.optimizer)
    setup.train(model, epochs=args.epochs, verbose=args.verbose)
    import numpy as np
    np.save(args.model,
            {'setup': setup.bean(with_data=False),
             'model': model.to_json(),
             'weights': model.get_weights()})
    if args.verbose:
        print("saved model", args.model)
