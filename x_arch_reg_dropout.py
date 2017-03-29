from nn_mlp import Setup
from conllu import load, write, validate
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dropout, Dense

# from keras import regularizers as reg
# from keras import initializers as init
# from keras import constraints as const


def new_model(setup,
              form_emb_reg=None,
              form_emb_const='unit_norm',
              upos_emb_dim=10,
              upos_emb_reg=None,
              upos_emb_const='unit_norm',
              inputs_dropout=0.0,
              hidden_units=200,
              hidden_reg=None,
              hidden_const=None,
              hidden_dropout=0.0,
              output_reg=None,
              output_const=None,
              optimizer='adamax'):
    form = Input(name='form', shape=(18, ), dtype=np.uint16)
    upos = Input(name='upos', shape=(18, ), dtype=np.uint8)
    feat = Input(name='feat', shape=(18 * len(setup.feat2idx), ))
    i = [form, upos, feat]
    form = Embedding(
        input_dim=len(setup.form2idx),
        input_length=18,
        output_dim=50,
        embeddings_initializer='zeros',
        embeddings_regularizer=form_emb_reg,
        embeddings_constraint=form_emb_const,
        name='form_emb')(form)
    upos = Embedding(
        input_dim=len(setup.upos2idx),
        input_length=18,
        output_dim=upos_emb_dim,
        embeddings_initializer='uniform',
        embeddings_regularizer=upos_emb_reg,
        embeddings_constraint=upos_emb_const,
        name='upos_emb')(upos)
    form = Flatten(name='form_flat')(form)
    upos = Flatten(name='upos_flat')(upos)
    o = Concatenate(name='inputs')([form, upos, feat])
    if inputs_dropout:
        o = Dropout(rate=inputs_dropout, name='inputs_dropout')(o)
    o = Dense(
        units=hidden_units,
        activation='tanh',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=hidden_reg,
        kernel_constraint=hidden_const,
        name='hidden')(o)
    if hidden_dropout:
        o = Dropout(rate=hidden_dropout, name='hidden_dropout')(o)
    o = Dense(
        units=len(setup.idx2tran),
        activation='softmax',
        kernel_initializer='glorot_uniform',
        kernel_regularizer=output_reg,
        kernel_constraint=output_const,
        name='output')(o)
    m = Model(i, o, name='darc')
    m.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    m.get_layer('form_emb').set_weights([setup.form_emb.copy()])
    # copy necessary ????
    return m


dev = list(load("./setups/grc_proiel-ud-dev.conllu"))

setup = Setup.load("./setups/grc_proiel-labeled.npy")

model = new_model(
    setup,
    # form_emb_reg=None,
    # form_emb_const='unit_norm',
    # upos_emb_dim=10,
    # upos_emb_reg=None,
    # upos_emb_const='unit_norm',
    inputs_dropout=0.5,
    # hidden_units=200,
    # hidden_reg=None,
    # hidden_const=None,
    # hidden_dropout=0.0,
    # output_reg=None,
    # output_const=None,
    # optimizer='adamax'
)

for epoch in range(10):
    setup.train(model, verbose=2)
    for sent in dev:
        setup.parse(model, sent)
    validate(dev)
    write(dev, "inputs_dropout.5_e{}.conllu"
          .format(epoch))
