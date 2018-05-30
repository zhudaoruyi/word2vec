# coding:utf-8

from __future__ import unicode_literals

from gensim.models.word2vec import Word2Vec
from datetime import timedelta
from glob import glob
import numpy as np
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


if __name__ == '__main__':
    start_time = time.time()
    # data_dir = '/home/abc/ssd/pzw/nlp/data/0523/word_sep_ht03/'
    data_dir = '/home/zhwpeng/abc/nlp/data/0324/word_sep/'
    if not os.path.exists('model/'):
        os.makedirs('model/')

    txt_dirs = list()
    for fold in glob(data_dir+'*'):
        # txt_dirs = txt_dirs + glob(fold+'/*.txt')
        txt_dirs = txt_dirs + glob(fold+'/*.txt')[:100]  # 本地小批量数据
    print "训练样本总数是{}".format(len(txt_dirs))
    np.random.shuffle(txt_dirs)

    sample_content = []
    for txt in txt_dirs:
        with codecs.open(txt, 'r', encoding='utf-8') as fr:
            content = fr.read()
        if content == ' ':
            continue
        content = content.split(' ')[:400]
        if len(content):
            # content = [word for word in content if len(word) > 1][:400]  # 分词中的单字去掉，取样本前400个词
            # print ' '.join(content)
            sample_content.append(content)
    print "训练样本总数是{}".format(len(sample_content))
    print 'take time {}'.format(get_time(start_time))

    # 训练词向量模型
    model = Word2Vec(sample_content, size=100, window=5, min_count=5, workers=4)
    # 保存词向量模型
    model.save('model/word2vec_v1.m')
    model.wv.save_word2vec_format('model/word2vec_v1.model', binary=False)
    print 'word2vec model saved successfully!'

    print 'take time {}'.format(get_time(start_time))
    if '公司' in model.wv:
        print '公司 的词向量是', model.wv['公司']
    if '公司研究' in model.wv:
        print '公司研究 的词向量是', model.wv['公司研究']
    if '策略' in model.wv:
        print '策略 的词向量是', model.wv['策略']
