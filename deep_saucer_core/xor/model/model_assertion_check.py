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
# Model loading script for Assertion Check used with DeepSaucer

## Requirement
Same as Assertion Check project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- xor
|       `-- model
|           `-- model_assertion_check.py @
`-- assertion_testing (_assertion_dir)
    `-- lib (_assertion_lib)
        `-- utils (_assertion_utils)
            `-- structutil.py (use NetworkStruct)
"""
import sys
import numpy as np
import tensorflow as tf

from pathlib import Path

_root_dir = Path(__file__).absolute().parent.parent.parent.parent
_assertion_dir = Path(_root_dir).joinpath('assertion_testing')
_assertion_lib = Path(_assertion_dir).joinpath('lib')
_assertion_utils = Path(_assertion_lib).joinpath('utils')

sys.path.append(str(_assertion_dir))
sys.path.append(str(_assertion_lib))
sys.path.append(str(_assertion_utils))

from structutil import NetworkStruct


def run_training(model_path):
    ns = NetworkStruct()
    
    X = np.array(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    Y = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])

    x = tf.placeholder(tf.float32, shape=[None, 3], name='XOR')
    ns.set_input(x, description='["value_1", "value_2", "value_3"]')
    
    t = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.truncated_normal([3, 16],), name='weight1')
    b = tf.Variable(tf.zeros([16]), name='bias1')
    h = tf.nn.relu(tf.matmul(x, W) + b, name='layer1')

    V = tf.Variable(tf.truncated_normal([16, 1], ), name='weight2')
    c = tf.Variable(tf.zeros([1]), name='bias2')
    y = tf.nn.sigmoid(tf.matmul(h, V) + c, name='layer2')

    round_value = tf.cast(tf.round(y), tf.int32)

    ns.set_output(node=round_value)

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

    ns.set_info_by_session(sess)
    ns.save(sess, str(model_path.joinpath('train')))

    with open(str(model_path.joinpath('vars_list.txt')),
              'w') as ws:
        ns.print_vars(ws=ws)
        
    print('Saved Model: {}'.format(str(model_path)))

    return sess


def model_load(downloaded_data):
    model_path = Path(downloaded_data, 'tf_xor_assertion')
    ckpt_path = model_path.joinpath('train')
    meta_path = Path(str(ckpt_path) + '.meta')

    if meta_path.exists():
        sess = tf.Session()
        saver = tf.train.import_meta_graph(str(meta_path))
        saver.restore(sess, str(ckpt_path))
        return sess

    else:
        sess = run_training(model_path)
        return sess
