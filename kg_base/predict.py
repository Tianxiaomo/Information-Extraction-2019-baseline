#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: predict.py
@time: 2019/4/2 13:26
@desc:
'''
import json
import codecs
from tqdm import tqdm
from keras.layers import *
from keras.models import Model
import keras.backend as K

def gpu_set(gpu_num):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF
    import os
    if isinstance(gpu_num, (list, tuple)):
        gpu_num = ','.join(str(i) for i in gpu_num)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print('GPU config done!')

gpu_set(2)


id2predicate, predicate2id = json.load(open('./datasets/all_50_schemas_me.json',encoding='utf-8'))
id2predicate = {int(i):j for i,j in id2predicate.items()}
id2char, char2id = json.load(open('./datasets/all_chars_me1.json',encoding='utf-8'))

char_size = 128
num_classes = len(id2predicate)


def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def seq_gather(x):
    """seq是[None, seq_len, s_size]的格式，
    idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
    最终输出[None, s_size]的向量。
    """
    seq, idxs = x
    idxs = K.cast(idxs, 'int32')
    batch_idxs = K.arange(0, K.shape(seq)[0])
    batch_idxs = K.expand_dims(batch_idxs, 1)
    idxs = K.concatenate([batch_idxs, idxs], 1)
    return K.tf.gather_nd(seq, idxs)


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)


def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return K.max(seq, 1)


t_in = Input(shape=(None,))
s1_in = Input(shape=(None,))
s2_in = Input(shape=(None,))
k1_in = Input(shape=(1,))
k2_in = Input(shape=(1,))
o1_in = Input(shape=(None,))
o2_in = Input(shape=(None,))

t, s1, s2, k1, k2, o1, o2 = t_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in

mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(t)
t = Embedding(len(char2id)+2, char_size)(t) # 0: padding, 1: unk
t = Dropout(0.25)(t)
t = Lambda(lambda x: x[0] * x[1])([t, mask])
t = Bidirectional(LSTM(char_size//2, return_sequences=True))(t)
t = Bidirectional(LSTM(char_size//2, return_sequences=True))(t)

t_max = Lambda(seq_maxpool)([t, mask])
t_dim = K.int_shape(t)[-1]

h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([t, t_max])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
ps1 = Dense(1, activation='sigmoid')(h)
ps2 = Dense(1, activation='sigmoid')(h)

subject_model = Model(t_in, [ps1, ps2]) # 预测subject的模型


k1 = Lambda(seq_gather, output_shape=(t_dim, ))([t, k1])
k2 = Lambda(seq_gather, output_shape=(t_dim, ))([t, k2])
k = Concatenate()([k1, k2])

h = Lambda(seq_and_vec, output_shape=(None, t_dim*2))([t, t_max])
h = Lambda(seq_and_vec, output_shape=(None, t_dim*4))([h, k])
h = Conv1D(char_size, 3, activation='relu', padding='same')(h)
h = Conv1D(char_size, 3, dilation_rate=2,activation='relu', padding='same')(h)
po1 = Dense(num_classes+1, activation='softmax')(h)
po2 = Dense(num_classes+1, activation='softmax')(h)

object_model = Model([t_in, k1_in, k2_in], [po1, po2]) # 输入text和subject，预测object及其关系

train_model = Model([t_in, s1_in, s2_in, k1_in, k2_in, o1_in, o2_in],
                    [ps1, ps2, po1, po2])

s1 = K.expand_dims(s1, 2)
s2 = K.expand_dims(s2, 2)

s1_loss = K.binary_crossentropy(s1, ps1)
s1_loss = K.sum(s1_loss * mask) / K.sum(mask)
s2_loss = K.binary_crossentropy(s2, ps2)
s2_loss = K.sum(s2_loss * mask) / K.sum(mask)

o1_loss = K.sparse_categorical_crossentropy(o1, po1)
o1_loss = K.sum(o1_loss * mask[:, :, 0]) / K.sum(mask)
o2_loss = K.sparse_categorical_crossentropy(o2, po2)
o2_loss = K.sum(o2_loss * mask[:, :, 0]) / K.sum(mask)

loss = 2.5 * (s1_loss + s2_loss) + (o1_loss + o2_loss)

train_model.add_loss(loss)
train_model.compile(optimizer='adam')
train_model.summary()


def extract_items(text_in):
    R = []
    _s = [char2id.get(c, 1) for c in text_in]
    _s = np.array([_s])
    _k1, _k2 = subject_model.predict(_s)
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    for i,_kk1 in enumerate(_k1):
        if _kk1 > 0.5:
            _subject = ''
            for j,_kk2 in enumerate(_k2[i:]):
                if _kk2 > 0.5:
                    _subject = text_in[i: i+j+1]
                    break
            if _subject:
                _k1, _k2 = np.array([i]), np.array([i+j])
                _o1, _o2 = object_model.predict([_s, _k1, _k2])
                _o1, _o2 = np.argmax(_o1[0], 1), np.argmax(_o2[0], 1)
                for i,_oo1 in enumerate(_o1):
                    if _oo1 > 0:
                        for j,_oo2 in enumerate(_o2[i:]):
                            if _oo2 == _oo1:
                                _object = text_in[i: i+j+1]
                                _predicate = id2predicate[_oo1]
                                R.append((_subject, _predicate, _object))
                                break
    return list(set(R))


# schemas = json.load(open('./datasets/all_50_schemas',encoding='utf-8'))
schema = {}

with open('datasets/all_50_schemas','r',encoding='utf-8') as f:
    for l in tqdm(f):
        sch = json.loads(l)
        # schemas.appead(a)
        schema[sch.get('predicate')] = sch.get('object_type') + "|" + sch.get('subject_type')

# schema = {}
# for sch in schemas:
#     schema[sch.get('predicate')] = sch.get('object_type') + "|" + sch.get('subject_type')

for i in range(5):
    print('kfold: %d' % i)
    train_model.load_weights('checkpoint/best_1dcnn_model_kfold{}.weights'.format(i))

    f_test = open('datasets/test1_data_postag.json','r',encoding='utf-8')
    text = f_test.readlines()

    with open('submit/1dcnn_sub_kfold{}.json'.format(i), 'w', encoding='utf-8') as f:
        for d in tqdm(text):
            sam = {}
            t = json.loads(d).get('text')
            sam['text'] = t
            targets = extract_items(t)
            spo_list = []
            for t in targets:
                spo = {}
                o,s = schema.get(t[1]).split('|')
                spo['object_type'] = o
                spo['predicate'] = t[1]
                spo['object'] = t[2]
                spo['subject_type'] = s
                spo['subject'] = t[0]
                spo_list.append(spo)
            sam['spo_list'] = spo_list
            json.dump(sam, f,ensure_ascii=False)
            f.write('\n')