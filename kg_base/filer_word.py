#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: filer_word.py
@time: 2019/4/7 18:01 
@desc:
'''
import json
from tqdm import tqdm
import pickle
import numpy as np
# words = []
# f = open('datasets/dev_data.json','r',encoding='utf-8')
# for l in tqdm(f.readlines()):
#     b = json.loads(l)
#     for w in b['postag']:
#         words.append(w['word'])
# f = open('datasets/train_data.json','r',encoding='utf-8')
# for l in tqdm(f.readlines()):
#     b = json.loads(l)
#     for w in b['postag']:
#         words.append(w['word'])
# f = open('datasets/test1_data_postag.json','r',encoding='utf-8')
# for l in tqdm(f.readlines()):
#     b = json.loads(l)
#     for w in b['postag']:
#         words.append(w['word'])
#
#
# print(len(words))
# print(len(set(words)))
# words = set(words)
f = open("datasets/words.pkl", 'rb')
words = pickle.load(f)
f.close()

f = open('../../Tencent_AILab_ChineseEmbedding.txt','r',encoding='utf-8')
vec = f.readlines()

# for line in vec:
#     s_s = line.split()
#     if s_s[0] in words:
#         vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])

vec = vec[1:]

import multiprocessing as mlp
from multiprocessing import Pool

# 任务数量
num = len(vec)
print(num)
# 子任务
def task(a, b):
    vocab_dic = {}
    for line in tqdm(vec[a:b]):
        try:
            s_s = line.split()
            if s_s == []:continue
            if s_s[0] in words:
                vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])
        except BaseException as e:
            print(e,s_s)
    print('sub task')
    return vocab_dic

# 线程池
p = Pool()
n_cpu = mlp.cpu_count()
split = num // n_cpu

result = []
for i in range(n_cpu):
    a = split * i
    if i == n_cpu - 1:
        b = num
    else:
        b = split * (i + 1)
    result.append(p.apply_async(task, args=(a, b)))

voc_dic = {}

for r in result:
    voc_dic = dict(voc_dic,**r.get())

p.close()
p.join()

file = open('vocab.pickle', 'wb')
pickle.dump(voc_dic, file)
file.close()