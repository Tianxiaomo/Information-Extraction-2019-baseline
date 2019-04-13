#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: Model.py
@time: 2019/4/8 13:35 
@desc:
'''
import keras.backend as K
from keras.layers import SpatialDropout1D,Lambda,Bidirectional,CuDNNGRU,Input,Embedding,Concatenate,GRU,CuDNNLSTM,LSTM
from keras.models import Model

from crf import CRF

def return_input(cfg):
    x_in = Input((cfg['maxlen'],), name='word')
    pos_tag_in = Input((cfg['maxlen'],), name='pos_tag')
    # py_in = Input((cfg['maxlen'],), name='pinyin')
    # radical_in = Input((cfg['maxlen'],), name='radical')
    # bound_in = Input((cfg['maxlen'],), name='bound')

    x = Embedding(cfg['vocab'], cfg['word_dim'], trainable=True, name='emb')(x_in)
    x = SpatialDropout1D(0.2)(x)

    pos_tag = Embedding(cfg['num_pg'], 16, name='embpos')(pos_tag_in)
    # bound = Embedding(cfg['num_bound'], 4, name='embbound')(bound_in)
    # pinyin = Embedding(cfg['num_pinyin'], 16)(py_in)
    # radical = Embedding(cfg['num_radical'], 16)(radical_in)
    x = Concatenate(axis=-1)([pos_tag, x])

    return x,{
        'word':x_in,
        'pos_tag':pos_tag_in,
    }


def crf_model(cfg):

    x, inputs = return_input(cfg)
    mask = Lambda(lambda x: K.cast(K.greater(x, 0), 'float32'))(inputs['word'])
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])

    x = Bidirectional(CuDNNLSTM(cfg['unit1'], return_sequences=True, name='gru1'), merge_mode='sum')(x)
    x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    x = SpatialDropout1D(0.3)(x)

    # x = Bidirectional(GRU(cfg['unit2'], return_sequences=True, name='gru2'), merge_mode='sum')(x)
    # x = Lambda(lambda x: x[0] * K.expand_dims(x[1], axis=-1))([x, mask])
    # x = SpatialDropout1D(0.15)(x)

    crf = CRF(cfg['num_tags'], sparse_target=True,name='crf')
    output = crf(x, mask=mask)

    model = Model(inputs=list(inputs.values()), outputs=[output])

    # multipliers = {
    #     'emb':0.1,
    #     'embbound':0.1,
    #     'embpos':0.1,
    #     # 'crf':cfg['lr_crf'],
    #     'gru1':cfg['lr_layer1']
    # }
    # model.compile(optimizer=M_Nadam(cfg['lr'],multipliers=multipliers), loss=crf.loss_function)
    return model