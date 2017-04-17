import src_ud2 as ud2
from src_setup import Setup
from keras.models import model_from_json
import numpy as np
import json


path = "/media/training-datasets/universal-dependency-learning/"
in_folder = path + "conll17-ud-trial-2017-03-19/"
# infolder = path + "conll17-ud-development-2017-03-19/"
out_folder = "????"

udpipe_model_folder = "????"
darc_model_folder = "????"


def read_meta(metadata):
    """-> lang, udfile, rawfile, outfile"""
    lang = metadata['ltcode']
    udfile = lang + "-udpipe.conllu"
    rawfile = metadata['rawfile']
    outfile = metadata['outfile']
    if lang not in ud2.treebanks:
        lang = metadata['lcode']
    if lang not in ud2.treebanks:
        lang = None
        # TODO: deal with unknown languages
    return lang, udfile, rawfile, outfile


def load_model(file):
    """-> Setup, keras.models.Model"""
    bean = np.load(file).item()
    setup = Setup(bean['setup'])
    model = model_from_json(bean['model'])
    model.set_weights(bean['weights'])
    return setup, model


with open("./lab/metadata.json") as file:
    metadata = json.load(file)
