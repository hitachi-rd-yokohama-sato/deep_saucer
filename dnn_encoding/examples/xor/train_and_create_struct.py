# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
"""
# 1. Automatic output tutorial of information
"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

from lib.utils.structutil import NetworkStruct

_proj_dir = Path(__file__).absolute().parent.parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_dir = Path(_proj_dir, 'examples')
_examples_xor = Path(_examples_dir, 'xor')
_examples_xor_model_dir = Path(_examples_xor, 'model')

_current = Path(os.getcwd()).absolute()

if __name__ == '__main__':
    # 2.
    ns = NetworkStruct()

    X = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

    x = tf.placeholder(tf.float32, shape=[None, 3], name='XOR')
    # 3.
    ns.set_input(placeholder=x, description='["value_1", "value_2", "value_3"]')

    t = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.truncated_normal([3, 16], ), name='weight1')
    b = tf.Variable(tf.zeros([16]), name='bias1')
    h = tf.nn.relu(tf.matmul(x, W) + b, name='layer1')
    # 4.
    ns.set_hidden(layer=h, weight=W, bias=b, description='["l1_0", "l1_1"]')

    V = tf.Variable(tf.truncated_normal([16, 1], ), name='weight2')
    c = tf.Variable(tf.zeros([1]), name='bias2')
    y = tf.nn.sigmoid(tf.matmul(h, V) + c, name='layer2')

    # 5.
    ns.set_output(node=y, weight=V, bias=c)

    cross_entropy = -tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # training
    for epoch in range(100000):
        sess.run(train_step, feed_dict={
            x: X,
            t: Y
        })

        if epoch % 10000 == 0:
            print('epoch:', epoch)

    # 6.
    ns.set_info_by_session(sess=sess)
    # 7.
    ns.save(sess=sess, path=str(_examples_xor_model_dir.joinpath('train')))
    # 8.
    # std out
    ns.print_vars()
    # File output
    with open(str(_examples_xor_model_dir.joinpath('vars_list.txt')), 'w') as ws:
        ns.print_vars(ws=ws)
