from nn_mlp import Setup
from conllu import load, write, validate

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from keras import regularizers as reg


def new_model(setup,
              upos_emb_dim=30,
              upos_emb_l1=-1,
              hidden_units=200,
              optimizer='adamax'):
    form = Input(name='form', shape=(18, ), dtype=np.uint16)
    upos = Input(name='upos', shape=(18, ), dtype=np.uint8)
    feat = Input(name='feat', shape=(18 * len(setup.feat2idx), ))
    i = [form, upos, feat]
    form = Embedding(
        name='form_emb',
        input_dim=len(setup.form2idx),
        output_dim=50,
        input_length=18)(form)
    upos = Embedding(
        name='upos_emb',
        embeddings_regularizer=reg.l1(upos_emb_l1)
        if 0.0 <= upos_emb_l1 else None,
        input_dim=len(setup.upos2idx),
        output_dim=upos_emb_dim,
        input_length=18)(upos)
    form = Flatten(name='form_flat')(form)
    upos = Flatten(name='upos_flat')(upos)
    o = Concatenate(name='inputs')([form, upos, feat])
    o = Dense(name='hidden', units=hidden_units, activation='tanh')(o)
    o = Dense(
        name='output', units=len(setup.idx2tran), activation='softmax')(o)
    m = Model(i, o, 'darc')
    m.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.get_layer('form_emb').set_weights([setup.form_emb.copy()])
    # copy necessary ????
    return m


dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

upos_emb_dim = 30
upos_emb_l1 = 0.01

model = new_model(
    setup,
    upos_emb_dim=upos_emb_dim,
    upos_emb_l1=upos_emb_l1,
    optimizer='adamax')

for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "./results/{}d-upos_{}l1_e{}.conllu"
          .format(upos_emb_dim, upos_emb_l1, epoch))
