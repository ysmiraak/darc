def parse_args():
    """-> argparse.Namespace"""
    import argparse
    parser = argparse.ArgumentParser(description="the parse is darc and full of errors.")
    parser.add_argument('--verbose', '-v', action='count', help="maximum verbosity: -v")
    parser.add_argument('--model', required=True, help="npy model file to load")
    parser.add_argument('--parse', required=True, nargs='+', help="conllu files to parse")
    parser.add_argument('--write', required=True, nargs='+', help="conllu files to write")
    args = parser.parse_args()
    if len(args.parse) != len(args.write):
        from sys import exit
        exit("the number of files to parse does not match the number of files to write")
    return args


def load_model(file):
    """-> Setup | SetupNoLemma, keras.models.Model"""
    import numpy as np
    bean = np.load(file).item()
    from setup import Setup
    setup = Setup(bean['setup'])
    from keras.models import model_from_json
    model = model_from_json(bean['model'])
    model.set_weights(bean['weights'])
    return setup, model


if '__main__' == __name__:
    args = parse_args()
    if args.verbose:
        print("loading", args.model, "....")
    setup, model = load_model(args.model)
    from conllu import load, save
    for parse, write in zip(args.parse, args.write):
        if args.verbose:
            print("parsing", parse, "....")
        save((setup.parse(sent, model) for sent in load(parse)), write)
        if args.verbose:
            print("written", write)
