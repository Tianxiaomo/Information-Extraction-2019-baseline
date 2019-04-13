#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: PyCharm
@file: train.py
@time: 2019/4/9 17:30 
@desc:
'''
# import pudb;pu.db
import json
from keras.optimizers import Nadam
from keras.callbacks import Callback
from tqdm import tqdm
import numpy as np
from itertools import product

import crf
from Model import crf_model
from dataset import Dataloader
from cfg import cfg

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

gpu_set(1)

def main(model,dataloader,val_dataloader):
    lr = 0.001
    model.compile(optimizer = Nadam(),
                  loss=crf.crf_loss,
                  metrics=[crf.crf_accuracy]
              )

    class Evaluate(Callback):
        def __init__(self):
            self.F1 = []
            self.best = 0.

        def on_epoch_end(self, epoch, logs=None):
            self.epoch = epoch
            f1, precision, recall = self.evaluate()
            self.F1.append(f1)
            if f1 > self.best:
                self.best = f1
                model.save_weights('best_model1.weights')
            print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))

        def evaluate(self):
            A, B, C = 1e-10, 1e-10, 1e-10
            with tqdm(total=len(val_dataloader),ncols=100,desc="Test Epoch:{}".format(self.epoch)) as pbar:
                for d in val_dataloader:
                    l = model.predict(d[0])
                    l_l = l.shape[1]
                    for l_s,d_s,t_s in zip(l,d[0]['word'],d[1][:,:,0]):
                        l_len = (d_s == 0).sum()
                        l_s = l_s.argmax(-1)
                        l_s = l_s[l_len:]
                        label = []
                        for i in np.where((l_s != 1) & (l_s % 3 == 1))[0]:
                            for j in range(i + 1,l_l-l_len):
                                if l_s[j] == l_s[i] + 1:continue
                                elif l_s[j] == l_s[i] + 2:
                                    label.append([i,j,l_s[i]//3])
                                    break
                                else:break

                        target = []
                        t_s = t_s[l_len:]
                        for i in np.where((t_s != 1) & (t_s % 3 == 1))[0]:
                            for j in range(i + 1,l_l-l_len):
                                if t_s[j] == t_s[i] + 1:continue
                                elif t_s[j] == t_s[i] + 2:
                                    target.append([i,j,t_s[i]//3])
                                    break
                                else:break

                        num = 0
                        for l,t in product(label,target):
                            if l==t :num+= 1

                        A += num
                        B += len(label)
                        C += len(target)

                    pbar.set_postfix({'f1':'{0:1.5f}'.format(2*A/(B+C))})  # 输入一个字典，显示实验指标
                    pbar.update(1)

            return 2 * A / (B + C), A / B, A / C

    evaluator = Evaluate()
    model.fit_generator(dataloader,
                              steps_per_epoch=len(dataloader),
                              epochs=100,
                              callbacks=[evaluator]
                              )
    # model.compile(optimizer=M_Nadam(cfg['lr'],multipliers=multipliers), loss=crf.loss_function)


if __name__ == '__main__':
    data = open('../datasets/train_data1.json','r',encoding='utf-8').readlines()
    val_data = open('../datasets/dev_data.json','r',encoding='utf-8').readlines()
    id2char, char2id = json.load(open('../datasets/all_chars_me1.json', encoding='utf-8'))
    id2entity,entity2id = json.load(open('../datasets/all_entity_type.json',encoding='utf-8'))
    dataloader = Dataloader(data,char2id,entity2id,cfg)
    val_dataloader = Dataloader(val_data,char2id,entity2id,cfg)

    net = crf_model(cfg)
    net.summary()
    # net.load_weights('best_model1.weights')
    main(net,dataloader,val_dataloader)

