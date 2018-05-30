# coding:utf-8

from __future__ import unicode_literals

from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from keras.utils import to_categorical
from datetime import timedelta
from glob import glob
import numpy as np
import datetime
import codecs
import time
import os


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def get_time(start_time):
    end_time = time.time()
    time_dif = start_time - end_time
    return timedelta(seconds=int(round(time_dif)))


def data_loader(file_dirs, labels, model, seq_length=400, size=100, classes=29, train_flag=True):
    X = list()
    y = list()
    unk = np.random.random(size)  # 未登录词，补零词，100维随机矩阵
    for txt in file_dirs:
        with codecs.open(txt, 'r', encoding='utf-8', errors='ignore') as fr:
            content = fr.read()
        content = content.split(' ')[:400]  # 不去掉单字，取样本前400个词
        if not len(content):
            continue
        # content = [word for word in content if len(word) > 1][:400]  # 分词中的单字去掉，取样本前400个词
        txt_array = list()
        for word in content:
            if word in model.wv:
                txt_array.append(model.wv[word])  # 训练好的词向量模型中，存在该词
            else:
                txt_array.append(unk)  # 未登录词，补零（随机矩阵）
        if len(txt_array) < seq_length:  # 文本长度不够的，在后面补零（随机矩阵），补零的一致补到序列长度为400
            for i in range(seq_length - len(txt_array)):
                txt_array.append(unk)
        X.append(np.array(txt_array))
        y.append(labels.index(txt.split('/')[-2]))  # 文本的路径可以看出其所在的文件夹，即类别
    X = np.array(X)
    y = to_categorical(y, num_classes=classes)
    if train_flag:
        x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        return x_train, y_train, x_val, y_val
    else:
        return X, y


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def read_category(categories):
    """读取分类目录，固定"""
    # categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = [native_content(x) for x in categories]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


if __name__ == '__main__':
    # 载入词向量模型
    model = Word2Vec.load('/home/zhwpeng/abc/text_classify/word2vec/model/word2vec_m2.m')
    print 'word2vec model loaded successfully!'
    # print model.wv.index2word
    print len(model.wv.index2word)
    for word in model.wv.index2word:
        print word
    # print model.wv.vector_size

    if 0:
        print "测试词向量模型..."
        for (word, prob) in model.wv.most_similar('报告'):
            print word, prob

        if '公司' in model.wv:
            print '公司 的词向量是', model.wv['公司']
        if '公司研究' in model.wv:
            print '公司研究 的词向量是', model.wv['公司研究']
        if '策略' in model.wv:
            print '策略 的词向量是', model.wv['策略']
    if 0:
        types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
                 'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
                 'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
                 'T004019003', 'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001', 'T004006005',
                 'OTHER']

        # data_dir = '/home/abc/ssd/pzw/nlp/data/0523/word_sep/'
        data_dir = '/home/zhwpeng/abc/nlp/data/0324/word_sep/'
        if not os.path.exists('model/'):
            os.makedirs('model/')

        txt_dirs = list()
        for fold in glob(data_dir + '*'):
            # txt_dirs = txt_dirs + glob(fold+'/*.txt')
            txt_dirs = txt_dirs + glob(fold + '/*.txt')[:100]  # 本地小批量数据
        print "训练样本总数是{}".format(len(txt_dirs))
        np.random.shuffle(txt_dirs)
        x_train, y_train, x_val, y_val = data_loader(txt_dirs, types, model)
        print "训练集x shape {},y shape {},验证集x shape {},y shape {}".format(
            x_train.shape, y_train.shape, x_val.shape, y_val.shape)
        # print x_train[:2], y_train[:2]

    if 0:
        print '查看词向量模型中的关键词...'
        with codecs.open('model/word2vec_m2.model', 'r', encoding='utf-8') as fr:
            cont = fr.read()
        cont = cont.splitlines()
        features = list()
        for line in cont:
            line = line.split(' ')
            for word in line:
                if not word.isalnum():
                    continue
                features.append(word)
        # print ' '.join(features)
        print len(features)


