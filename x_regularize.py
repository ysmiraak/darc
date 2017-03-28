from nn_mlp import Setup
from conllu import load, write, validate

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras.regularizers import l2

upos_emb_dim = 10
hidden_units = 200


def new_model(setup, hidden_l2=-1, output_l2=-1, emb_l2=-1, optimizer='adamax'):
    """-> keras.models.Model

    feature: Feature

    w2v: gensim.models.keyedvectors.KeyedVectors

    """
    form = Input(name='form', shape=(18, ), dtype=np.uint16)
    upos = Input(name='upos', shape=(18, ), dtype=np.uint8)
    feat = Input(name='feat', shape=(18 * len(setup.feat2idx), ))
    i = [form, upos, feat]
    form = Embedding(
        name='form_emb',
        embeddings_regularizer=l2(emb_l2) if 0.0 <= emb_l2 else None,
        input_dim=len(setup.form2idx),
        output_dim=50,
        input_length=18)(form)
    upos = Embedding(
        name='upos_emb',
        embeddings_regularizer=l2(emb_l2) if 0.0 <= emb_l2 else None,
        input_dim=len(setup.upos2idx),
        output_dim=upos_emb_dim,
        input_length=18)(upos)
    form = Flatten(name='form_flat')(form)
    upos = Flatten(name='upos_flat')(upos)
    o = Concatenate(name='inputs')([form, upos, feat])
    o = Dense(
        name='hidden',
        units=hidden_units,
        kernel_regularizer=l2(hidden_l2) if 0.0 <= hidden_l2 else None,
        activation='tanh')(o)
    o = Dense(
        name='output',
        units=len(setup.idx2tran),
        kernel_regularizer=l2(output_l2) if 0.0 <= output_l2 else None,
        activation='softmax')(o)
    # TODO: add regularization
    m = Model(i, o, 'darc')
    m.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.get_layer('form_emb').set_weights([setup.form_emb.copy()])
    # copy necessary ????
    return m


ud_path = "/data/ud-treebanks-conll2017/"

setup = Setup.load("./setups/grc_proiel-labeled.npy")

dev = list(load(ud_path + "/UD_Ancient_Greek-PROIEL/grc_proiel-ud-dev.conllu"))


def experiment(filename, *args, **kwargs):
    model = new_model(setup, *args, **kwargs)
    for epoch in range(10):
        setup.train(model, verbose=2)
        for sent in dev:
            setup.parse(model, sent)
        validate(dev)
        write(dev, filename.format(epoch))


experiment("./results/adam_hidden.001_e{}.conllu",
           hidden_l2=0.001, optimizer='adam')
experiment("./results/adamax_hidden.001_e{}.conllu",
           hidden_l2=0.001, optimizer='adamax')
experiment("./results/adam_hidden.001_output.001_e{}.conllu",
           hidden_l2=0.001, output_l2=0.001, optimizer='adam')
experiment("./results/adamax_hidden.001_output.001_e{}.conllu",
           hidden_l2=0.001, output_l2=0.001, optimizer='adamax')
experiment("./results/adam_hidden.001_output.001_emb.001_e{}.conllu",
           hidden_l2=0.001, output_l2=0.001, emb_l2=0.001, optimizer='adam')
experiment("./results/adamax_hidden.001_output.001_emb.001_e{}.conllu",
           hidden_l2=0.001, output_l2=0.001, emb_l2=0.001, optimizer='adamax')
