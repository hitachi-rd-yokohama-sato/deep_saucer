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
# Model loading script for SafeCV used with DeepSaucer

## Requirement
Same as SafeCV project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- mnist
|       `-- model
|           `-- model_safecv.py @
`-- safecv (_safecv_dir)
    `-- Examples
        `-- MNIST-Example (_safecv_examples_mnist_dir)
            `-- cw-attacks-mnist.weights
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
import tensorflow as tf

import sys
import os

_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_safecv_dir = os.path.join(_root_dir, 'safecv')
_safecv_examples_mnist_dir = os.path.join(
    _safecv_dir, 'Examples', 'MNIST-Example')


def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                   logits=predicted / 1)


def model_load(downloaded_data):
    
    model = Sequential()

    model.add(Conv2D(32, 3, 3, input_shape=(1, 28, 28),  dim_ordering="th"))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])

    model.load_weights(
        os.path.join(_safecv_examples_mnist_dir, "cw-attacks-mnist.weights"))

    return model
