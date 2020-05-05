# -*- coding: utf-8 -*-

"""
# @Software: PyCharm
# DESC : 基于bert的等价性问题判别
"""
import zipfile
import codecs
import pandas as pd
import numpy as np
import logging
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from tqdm import tqdm

pd.set_option('display.max_columns', None)

extracting = zipfile.ZipFile('../data/External/RoBERTa-large-pair.zip')
extracting.extractall("../data/External/RoBERTa-large-pair")

# bert预训练模型路径
base = "../data/External/RoBERTa-large-pair/"

config_path = base + 'bert_config.json'
checkpoint_path = base + 'bert_model.ckpt'
dict_path = base + 'vocab.txt'

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

max_len = 40
epochs = 5
batch_size = 16
model_type = 0


def read_data():
    x_data = []
    y_data = []

    train = pd.read_csv("../data/Dataset/train.csv")

    query1 = train["query1"]
    query2 = train["query2"]
    label = train["label"]
    for qu1, qu2, lab in zip(query1, query2, label):
        x_data.append([qu1, qu2])
        y_data.append(int(lab))
    return np.array(x_data), np.array(y_data)

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


def seq_padding(X, padding=0, maxlen=None):
    if maxlen is None:
        L = [len(x) for x in X]
        ML = max(L)
    else:
        ML = maxlen
    return np.array([
        np.concatenate([x[:ML], [padding] * (ML - len(x))]) if len(x[:ML]) < ML else x for x in X
    ])


# maxpool时消除mask部分的影响
def seq_maxpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return MaxPool1D(padding='same')(seq)
    # return K.max(seq, keepdims=True)


def seq_avgpool(x):
    """seq是[None, seq_len, s_size]的格式，
    mask是[None, seq_len, 1]的格式，先除去mask部分，
    然后再做maxpooling。
    """
    seq, mask = x
    seq -= (1 - mask) * 1e10
    return AvgPool1D(padding='same')(seq)


class data_generator:
    def __init__(self, data, max_len, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.max_len = max_len
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text1 = d[0][:max_len]
                text2 = d[1][:max_len]
                x1, x2 = tokenizer.encode(first=text1, second=text2)
                y = d[2]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    # print(X1.shape, X2.shape)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


# 样本不均衡时使用的损失函数focal_loss
def focal_loss_fixed(y_true, y_pred):
    # y_pred = K.sigmoid(y_pred)
    gamma = 2.0
    alpha = 0.25
    epsilon = 1e-6
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + epsilon)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + epsilon))


def trian_model_bert():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    # print(x.shape)
    x = Lambda(lambda x: x[:, 0])(x)  # 只取cls用于分类
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

    x1, x2 = x1_in, x2_in
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
    # print(x.shape,  t.shape)
    # x = Lambda(lambda x: x[:, 0])(x)  #只取cls用于分类
    c = concatenate([x, t], axis=-1)
    c = Lambda(lambda c: c[:, 0])(c)
    p = Dense(1, activation='sigmoid')(c)

    model = Model([x1, x2], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=['accuracy']
    )
    model.summary()
    return model


def get_mode_type(model_type=1):
    trian_model = ''
    if model_type == 0:
        trian_model = trian_model_bert()
    elif model_type == 1:
        trian_model = trian_model_bertlstmgru()
    return trian_model, model_type


def predict(dev_set):
    pbar = tqdm()
    result = []
    result1 = []
    for index, row in dev_set.iterrows():
        query1 = str(row["query1"])
        query2 = str(row["query2"])
        x1, x2 = tokenizer.encode(first=query1, second=query2)
        x1 = x1[:max_len]
        x2 = x2[:max_len]
        tmp_result = model.predict([np.array([x1]), np.array([x2])])
        result_label = tmp_result[0][0]  # 直接取值，例如[[0.99]]取值为0.99
        result.append(result_label)  # 此拼接为横向拼接

        if result_label > 0.5:
            result1.append(1)
        else:
            result1.append(0)
        pbar.update(1)

    print("该轮验证集得分为{}".format(accuracy_score(result1, real)))
    pbar.close()
    return np.array(result)


dev_set = pd.read_csv("../data/Dataset/dev.csv")
dev_query1 = dev_set["query1"].values
dev_query2 = dev_set["query2"].values
oof_dev = [0.0] * (len(dev_set))
oof_dev = np.array(oof_dev)
real = dev_set["label"].tolist()

if __name__ == '__main__':

    train_x, train_y = read_data()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(skf.split(train_x, train_y)):
        logger.info('================     fold {}        ==============='.format(fold))

        x1 = train_x[train_index]
        y = train_y[train_index]
        val_x1 = train_x[valid_index]
        val_y = train_y[valid_index]

        train_data = np.column_stack((x1, y))  # 为矩阵格式
        valid_data = np.column_stack((val_x1, val_y))

        train_D = data_generator(train_data, max_len, batch_size)
        valid_D = data_generator(valid_data, max_len, batch_size)
        model, model_type = get_mode_type(model_type)
        model_weight_filepath = "../user_data/model_data/" + str(0) + "_" + str(fold) + ".weights"
        earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=2)  # 若2个epoch没有提高则early_stopping
        reducelronplateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.8,
                                              patience=1)  # val_acc在1个epoch没有提高，则学习率下降0.8 （new_lr） = lr * factor
        checkpoint = ModelCheckpoint(filepath=model_weight_filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True, save_weights_only=True, mode='max',
                                     period=1)  # save_best_only：当设置为True时，监测值有改进时才会保存当前的模型；在这里若epochs有改进，则覆盖前一个。

        model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),  # 从generator产生的步骤的总数（样本批次总数）。通常情况下，应该等于数据集的样本数量除以批量的大小。
            epochs=epochs,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=[earlystopping, reducelronplateau, checkpoint])

        model.load_weights(model_weight_filepath)
        oof_dev += predict(dev_set)
        del model
        K.clear_session()

    oof_dev = oof_dev / 5

    for i in range(len(oof_dev)):
        if oof_dev[i] < 0.5:
            oof_dev[i] = 0
        else:
            oof_dev[i] = 1
    oof_dev = oof_dev.tolist()
    print("最终验证集得分为：")
    print(accuracy_score(oof_dev, real))






