# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 100  # 词向量维度
    seq_length = 400  # 序列长度
    num_classes = 29  # 类别数
    num_filters = 256  # 卷积核数目
    # kernel_size = 5  # 卷积核尺寸
    filter_sizes = [3, 5, 7]
    # vocab_size = 5000  # 词汇表达
    vocab_size = 97639

    hidden_dim = 512  # 全连接层神经元

    dropout_keep_prob = 1.  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 10  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.float32, [None, self.config.seq_length, self.config.embedding_dim], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # with tf.device('/cpu:0'), tf.name_scope("cnn"):
        #     # CNN layer
        #     conv = tf.layers.conv1d(self.input_x, self.config.num_filters, self.config.kernel_size, name='conv')
        #     # global max pooling layer
        #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("conv_maxpool_0"):
            # conv layer
            conv_3 = tf.layers.conv1d(self.input_x, self.config.num_filters, self.config.filter_sizes[0], name='conv_3')
            # relu layer
            h_3 = tf.nn.relu(conv_3, name='relu')
            # global max pooling layer
            pooling_3 = tf.reduce_max(h_3, reduction_indices=[1], name='global_max_pooling')

        with tf.name_scope("conv_maxpool_1"):
            # conv layer
            conv_5 = tf.layers.conv1d(self.input_x, self.config.num_filters, self.config.filter_sizes[1], name='conv_5')
            # relu layer
            h_5 = tf.nn.relu(conv_5, name='relu')
            # global max pooling layer
            pooling_5 = tf.reduce_max(h_5, reduction_indices=[1], name='global_max_pooling')

        with tf.name_scope("conv_maxpool_2"):
            # conv layer
            conv_7 = tf.layers.conv1d(self.input_x, self.config.num_filters, self.config.filter_sizes[2], name='conv_7')
            # relu layer
            h_7 = tf.nn.relu(conv_7, name='relu')
            # global max pooling layer
            pooling_7 = tf.reduce_max(h_7, reduction_indices=[1], name='global_max_pooling')

        # num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        pooled_concat = tf.concat([pooling_3, pooling_5, pooling_7], axis=1, name='pooled_concat')
        # pooled_concat_flat = tf.reshape(pooled_concat, shape=[-1, num_filters_total], name='pooled_concat_flat')
        # print(self.pooling_3)
        # print(self.pooled_concat)
        # print(pooled_concat_flat)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            # fc = tf.layers.dense(self.pooled_concat, self.config.hidden_dim, name='fc1')
            fc = tf.layers.dense(pooled_concat, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
