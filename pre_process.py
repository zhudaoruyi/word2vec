# coding=utf-8
# ------本脚本从txt中获取文本内容，并合并行段落、分词、去除停用词------

from __future__ import unicode_literals

import re
import os
import sys
import json
import jieba
import jieba.posseg as pseg
import codecs
import datetime
import numpy as np
from tqdm import tqdm
from glob import glob
from pyhanlp import *

nature = 'n nr ns nt nz nl ng t tg s f vi vg a ad an ag d i'
nature2 = 'eng m p pba pbei c cc u e y o w wkz wky wyz wyy wj ww wt wd wf wn wm ws wp wb wh'
# jieba.load_userdict('abcChinese_j0523.txt')


def current_time():
    """
    获取当前时间：年月日时分秒
    :return:
    """
    ct = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    return ct


def separate_words_hanlp(input_text, stopwords=None):
    out_txt = list()
    # seg_list = jieba.cut(input_text)
    # seg_list = pseg.cut(input_text)
    # attachThreadToJVM()

    seg_list = HanLP.segment(input_text)
    for term in seg_list:
    # for word in seg_list:
    # for word, flag in seg_list:
        # print word, flag
        if term.word not in stopwords and term.word.strip() != '' and term.word.lower().isalpha():
        # if flag not in nature2.encode('utf-8').split(' ') and len(word.decode('utf-8')) > 1:
        # if len(word.decode('utf-8')) > 1:
        #     print word, flag
            out_txt.append(term.word)
    return ' '.join(out_txt[:400])

def separate_words_jieba(input_text, stopwords=None):
    out_txt = list()
    seg_list = jieba.cut(input_text)
    # seg_list = pseg.cut(input_text)
    # seg_list = HanLP.segment(input_text)
    for word in seg_list:
        if word not in stopwords and word.strip() != '' and word.lower().isalpha():
            out_txt.append(word)
    return ' '.join(out_txt[:400])

def separate_words_pseg(input_text, stopwords=None):
    out_txt = list()
    # seg_list = jieba.cut(input_text)
    seg_list = pseg.cut(input_text)
    # seg_list = HanLP.segment(input_text)
    # for term in seg_list:
    # for word in seg_list:
    for word, flag in seg_list:
        # print word, flag
        # if flag in nature.split(' ') and len(word) > 1 and word not in stopwords:
        # if flag not in nature2.encode('utf-8').split(' ') and len(word.decode('utf-8')) > 1:
        # if len(word.decode('utf-8')) > 1:
        #     print word, flag
        #if word not in stopwords and word.strip() != '' and ( word.lower().isalnum() or not word.lower().isdigit() or word.lower().isalpha()):
        if word not in stopwords and word.strip() != '' and word.lower().isalpha():
            out_txt.append(word.strip())
    tmp = ' '.join(out_txt[:400])
    return tmp

def pre_process(full_text, stopwords=None):
    full_text = full_text.replace(u'\n', '')
    # full_text = re.sub(u"[：（）“”，■？、！…,/'《》<>!?_—=()。【】￭＃]+", "", full_text)  # 去掉中文标点符号
    # full_text = re.sub("[-`~_+=.:@]+", "", full_text)  # 去掉英文标点符号
    # print full_text
    # full_text1 = separate_words_hanlp(full_text, stopwords=stopwords)
    # print 'HanLP 分词结果 \n', full_text1
    full_text2 = separate_words_jieba(full_text, stopwords=stopwords)
    print 'jieba 分词结果 \n\n', full_text2
    # full_text3 = separate_words_pseg(full_text, stopwords=stopwords)
    # print 'jieba.posseg 分词结果 \n\n', full_text3
    return full_text2


if __name__ == '__main__':
    jieba.load_userdict('abcChinese_j052301.txt')
    # base_dir = '/home/abc/ssd/pzw/nlp/data/'
    base_dir = '/home/zhwpeng/abc/text_classify/data/0412/'
    # 原始数据存放位置
    # base_data_dir = os.path.join(base_dir, 'train_data/')
    base_data_dir = os.path.join(base_dir, 'raw/train_data/')
    # 分词后的数据存放位置
    # separated_word_file_dir = os.path.join('/home/abc/ssd/pzw/nlp/data/0523/', 'word_sep/')
    separated_word_file_dir = os.path.join('/home/zhwpeng/abc/text_classify/word2vec/', 'word_sep/')
    if not os.path.exists(separated_word_file_dir):
        os.makedirs(separated_word_file_dir)

    types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
             'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
             'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
             'T004019003', 'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001', 'T004006005',
             'OTHER']

    stop_words_file = "stop_words_v0520.txt"
    stop_words = set()
    # 获取停用词
    with codecs.open(stop_words_file, 'r', encoding='utf-8') as fi:
        for line in fi.readlines():
            stop_words.add(line.strip())
    # print stopwords

    if 1:
        for ty in tqdm(types):
            txt_dirs = glob(base_data_dir + ty + '/*.txt')
            output_dir = os.path.join(separated_word_file_dir, ty)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for tk, txt_dir in enumerate(txt_dirs):
                # print txt_dir
                with codecs.open(txt_dir, 'r', encoding='utf-8', errors='ignore') as f:
                    r = f.read()
                # print r
                pre_process(r, stopwords=stop_words)
                fulltext = pre_process(r, stopwords=stop_words)
                # print fulltext
                with codecs.open(output_dir + '/' + txt_dir.split('/')[-1], 'w',
                                 encoding='utf-8') as fw:
                    fw.write(fulltext)

    if 0:
        with codecs.open(
                # '/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/T004005001/55910.txt',
                # '/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/T004004003/7342.txt',
                 '/home/zhwpeng/abc/text_classify/data/0412/raw/train_data/D001002001/2007641.txt',
                'r', encoding='utf-8', errors='ignore') as fr:
            content = fr.read()
        # print content
        full_text = pre_process(content, stopwords=stop_words)
        # print full_text
