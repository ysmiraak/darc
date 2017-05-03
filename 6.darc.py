import src_ud2 as ud2
import src_conllu as conllu
from src_setup import Setup
import json


task_path = "/media/training-datasets/universal-dependency-learning/conll17-ud-trial-2017-03-19/"
# "/media/training-datasets/universal-dependency-learning/conll17-ud-development-2017-03-19/"

darc_path = "/home/darc/darc/"
udpipe_parse_path = darc_path + "conll17/udpipe_parse/"
system_model_path = darc_path + "conll17/system_model/"
system_parse_path = darc_path + "conll17/system_parse/"


if '__main__' == __name__:

    with open(task_path + "metadata.json") as file:
        metadata = json.load(file)

    for task in metadata:
        outfile = task['outfile']
        lang = task['ltcode']
        if (lang not in ud2.treebanks) and (lang not in ud2.surprise):
            lang = task['lcode']

        print("loading model", lang, "....")
        setup, model = Setup.load("{}{}.npy".format(system_model_path, lang), with_model=True)
        conllu.save((setup.parse(model, sent) for sent in conllu.load(
            udpipe_parse_path + outfile)),
                    system_parse_path + outfile)
        print("written", outfile)
