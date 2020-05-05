# -*- coding: utf-8 -*-

"""
# @Software: PyCharm
# DESC : 基于bert的等价性问题判别
"""

import os
import zipfile
import codecs
import pandas as pd
import numpy as np
import csv
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from sklearn.metrics import f1_score, accuracy_score
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tqdm import tqdm

pd.set_option('display.max_columns', None)

# bert预训练模型路径
base = "../data/External/RoBERTa-large-pair/"

config_path = base + 'bert_config.json'
checkpoint_path = base + 'bert_model.ckpt'
dict_path = base + 'vocab.txt'

maxlen=40

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R

tokenizer = OurTokenizer(token_dict)




def trian_model_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    #print(x.shape)
    x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


def trian_model_bertlstmgru():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x1, x2 =x1_in, x2_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
    x = bert_model([x1, x2])
    t = Dropout(0.1)(x)
    t = Bidirectional(LSTM(80, recurrent_dropout=0.1, return_sequences=True))(t)
    t = Bidirectional(GRU(80, recurrent_dropout=0.1, return_sequences=True))(t)
    t = Dropout(0.4)(t)
    t = Dense(160)(t)
    # t_maxpool = Lambda(seq_maxpool)([t, mask])
    # t_maxpool = MaxPool1D()(t)
    # t_avgpool = Lambda(seq_avgpool)([t, mask])
    # t_ = concatenate([t_maxpool, t_avgpool], axis=-1)
    print(x.shape,  t.shape)
    # x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    c = concatenate([x, t], axis=-1)
    c = Lambda(lambda c: c[:, 0])(c)
    p = Dense(1, activation='sigmoid')(c)

    model = Model([x1, x2], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(2e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model

def get_mode_type(model_type = 0):
    trian_model = ''
    if model_type == 0:
        trian_model = trian_model_bert()
    elif model_type == 1:
        trian_model = trian_model_bertlstmgru()
    return trian_model, model_type


def test():
    pbar = tqdm()
    result = []
    result1 = []
    for index, row in test_set.iterrows():
        content1 = str(row["query1"])
        content2 = str(row["query2"])
        x1, x2 = tokenizer.encode(first=content1, second=content2)
        x1 = x1[:maxlen]
        x2 = x2[:maxlen]
        tmp_result = model.predict([np.array([x1]), np.array([x2])])
        result_label = tmp_result[0][0]
        result.append(result_label)

        result_label1 = tmp_result[0][0]
        if result_label1 > 0.5:
            result_label1 = 1
        else:
            result_label1 = 0
        result1.append(result_label1)

        pbar.update(1)

    pbar.close()

    return np.array(result)


test_set = pd.read_csv("../data/Dataset/test.csv")
submit = test_set
submit=submit.drop(["category"], axis=1)
submit=submit.drop(["query1"], axis=1)
submit=submit.drop(["query2"], axis=1)
oof_test = [0.0] * (len(test_set))
oof_test = np.array(oof_test)

if __name__ == '__main__':


    model, model_type = get_mode_type(0)
    
    model.load_weights("../user_data/model_data/0_0.weights")
    oof_test += test()*0.02
    model.load_weights("../user_data/model_data/0_1.weights")
    oof_test += test()*0.02
    model.load_weights("../user_data/model_data/0_2.weights")
    oof_test += test()*0.02
    model.load_weights("../user_data/model_data/0_3.weights")
    oof_test += test()*0.02
    model.load_weights("../user_data/model_data/0_4.weights")
    oof_test += test()*0.02
   
    model.load_weights("../user_data/model_data/1_0.weights")
    oof_test += test()*0.10
    model.load_weights("../user_data/model_data/1_1.weights")
    oof_test += test()*0.10
    model.load_weights("../user_data/model_data/1_2.weights")
    oof_test += test()*0.10
    model.load_weights("../user_data/model_data/1_3.weights")
    oof_test += test()*0.10
    model.load_weights("../user_data/model_data/1_4.weights")
    oof_test += test()*0.10
    
    model.load_weights("../user_data/model_data/2_0.weights")
    oof_test += test()*0.08
    model.load_weights("./user_data/model_data/2_1.weights")
    oof_test += test()*0.08
    model.load_weights("../user_data/model_data/2_2.weights")
    oof_test += test()*0.08
    model.load_weights("../user_data/model_data/2_3.weights")
    oof_test += test()*0.08
    model.load_weights("../user_data/model_data/2_4.weights")
    oof_test += test()*0.08

    
    labels = []
    for i in range(len(oof_test)):
        if oof_test[i] < 0.5:
            labels.append(0)
        else:
            labels.append(1)

    submit["label"] = labels
    submit.to_csv("../prediction_result/result.csv", index = False)










