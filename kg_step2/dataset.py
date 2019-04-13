#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: dataset.py
@time: 2019/4/8 14:18 
@desc:
'''
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

import json
import re
import math

class Dataloader(Sequence):
    def __init__(self,data,char2id,entity2id,cfg,train=True,onehot=False):
        self.data = np.asarray(data)
        self.data_len = len(data)
        self.char2id = char2id
        self.entity2id = entity2id
        self.idxs = list(range(len(self.data)))

        self.onehot = onehot
        self.num_tags = cfg['num_tags']
        np.random.shuffle(self.idxs)
        if train == False:
            self.batch_size = cfg['val_batch_size']
        else:
            self.batch_size = cfg['train_batch_size']

    def __len__(self):
        return math.ceil(self.data_len/self.batch_size)

    def __pos2id__(self,text,label,en,tp):
        eid = tp
        # w = [j.start() for j in re.finditer(text,en)]
        # label = [1]*len(text) # 0,mask,1,other,eid*3+2:start,eid*3+3:in,eid*3+4:end
        if '+' in en or '\\' in en or '*' in en or ')' in en or '(' in en or '?' in en or '[' in en:
            en1 = en.replace('+','\+')
            en1 = en1.replace('\\','\\\\')
            en1 = en1.replace('*','\*')
            en1 = en1.replace(')','\)')
            en1 = en1.replace('(','\(')
            en1 = en1.replace('?','\?')
            en1 = en1.replace('[','\[')

        else:
            en1 = en
        for j in re.finditer(en1,text):
            loc = j.start()
            label[loc] = eid*3 +1
            if len(en) == 2:
                label[loc+1] = eid*3 + 3
            elif len(en) == 3:
                label[loc+1] = eid*3 + 2
                label[loc+2] = eid*3 + 3
            else:
                label[loc+1:loc+len(en)-1] = eid*3 + 2
                label[loc+len(en)-1] = eid*3 + 3

        return label

    def __getitem__(self, item):
        data_label = self.data[self.idxs[item*self.batch_size:(item+1)*self.batch_size]]
        T = []
        P = []
        L = []
        for d_l in data_label:
            d_l = json.loads(d_l)
            T.append([self.char2id.get(i,1) for i in d_l['text']])
            # label = [1]*len(d_l['text'])
            label = np.ones(len(d_l['text']))
            for i in d_l['spo_list']:
                try:
                    label = self.__pos2id__(d_l['text'],label,i['object'],self.entity2id.get(i['object_type']))
                    label = self.__pos2id__(d_l['text'],label,i['subject'],self.entity2id.get(i['subject_type']))
                except BaseException as e:
                    print(e,d_l)
            L.append(label)
        T = self.seq_padding(T)
        L = self.seq_padding(L,value=-1)
        P = np.zeros_like(T)
        D = {'word':T,
             'pos_tag':P}

        if self.onehot:
            L = np.eye(len(L), dtype='float32')[L]
        else:
            L = np.expand_dims(L, -1)
        return D,L

    def seq_padding(self,X,value=0):
        L = [len(x) for x in X]
        ML = max(L)
        return pad_sequences(X,ML,value=value) # left padding

    def on_epoch_end(self):
        np.random.shuffle(self.idxs)


if __name__ == '__main__':
    data = open('../datasets/train_data.json','r',encoding='utf-8').readlines()
    id2char, char2id = json.load(open('../datasets/all_chars_me1.json', encoding='utf-8'))
    id2entity,entity2id = json.load(open('../datasets/all_entity_type.json',encoding='utf-8'))
    import cfg
    dataloader = Dataloader(data,char2id,entity2id,cfg.cfg)
    for i in range(len(dataloader)):
        # if i == 1352:
        #     print(i)
        d,l = dataloader.__getitem__(i)
    # print(d,l)