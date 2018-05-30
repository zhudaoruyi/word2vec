# -*- coding: utf-8 -*-

import os
import sys
import time
from glob import glob
from datetime import timedelta

import numpy as np
import tensorflow as tf
from sklearn import metrics
from gensim.models.word2vec import Word2Vec

from cnn_model import TCNNConfig, TextCNN
from data_loader import data_loader, batch_iter, read_category

base_dir = ''
save_dir = 'model/textcnn'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")

    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train, x_val, y_val = data_loader(txt_dirs, types, word2vec_model)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def model_tes_t(test_txt_dirs, train_flag=False):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = data_loader(test_txt_dirs, types, word2vec_model, train_flag=train_flag)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    types = ['NOTICE', 'T004001001', 'T004001002', 'D001002001', 'D001002002', 'D001002003', 'T004021008',
             'D001003001', 'D001003002', 'D001003003', 'T004022018', 'D001004001', 'D001004002', 'D001004003',
             'T004023007', 'T004004002', 'T004004005', 'T004004001', 'T004004004', 'T004004003', 'T004019001',
             'T004019003', 'T004009001', 'T004009002', 'T004005001', 'T004005002', 'T004006001', 'T004006005',
             'OTHER']

    # 载入词向量模型
    print "载入词向量模型..."
    word2vec_model = Word2Vec.load('model/word2vec_m2.m')
    # word2vec_model = Word2Vec.load('/home/zhwpeng/abc/text_classify/word2vec/model/word2vec_m2.m')
    print 'word2vec model loaded successfully!'

    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python train_cnn.py [train / test]""")

    print "配置CNN模型..."
    config = TCNNConfig()
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        print "开始训练..."
        print "载入训练样本..."
        # data_dir = '/home/abc/ssd/pzw/nlp/data/0523/word_sep/'
        data_dir = '/home/zhwpeng/abc/nlp/data/0324/word_sep/'
        txt_dirs = list()
        for fold in glob(data_dir + '*'):
            # txt_dirs = txt_dirs + glob(fold+'/*.txt')
            txt_dirs = txt_dirs + glob(fold + '/*.txt')[:1]  # 本地小批量数据
        print "训练样本总数是{}".format(len(txt_dirs))
        np.random.shuffle(txt_dirs)

        train()
    else:
        print "开始测试..."
        test_txt_dirs = list()
        test_data_dir = '/home/abc/ssd/pzw/nlp/data/0523/word_sep_test/'
        # test_data_dir = '/home/zhwpeng/abc/nlp/data/0324/word_sep_test/'
        for fold in glob(test_data_dir + '*'):
            test_txt_dirs = test_txt_dirs + glob(fold + '/*.txt')
        print "测试集样本总数是{}".format(len(test_txt_dirs))
        np.random.shuffle(test_txt_dirs)

        categories, cat_to_id = read_category(types)
        model_tes_t(test_txt_dirs, train_flag=False)
