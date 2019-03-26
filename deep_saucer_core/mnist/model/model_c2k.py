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
# Model loading script used with DeepSaucer

## Directory Structure

Any Directory
`-- DeepSaucer
    |-- downloaded_data (downloaded_path)
    |   `-- mnist_chainer (dl_dir)
    |       `-- chainer_model_mnist.npz (param_path)
    `-- mnist
        `-- model
            `-- model_c2k.py @
"""

import sys

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import pathlib


def model_load(downloaded_path):
    """
    Load mnist model
    Read parameters of model created by Chainer and generate model for Keras
    :return:
    """
    dl_dir = pathlib.Path(downloaded_path, 'mnist_chainer')
    param_path = dl_dir.joinpath('chainer_model_mnist.npz')

    model = _c2k_mnist(str(param_path))

    return model


def _c2k_mnist(param_path):
    """
    Chainer -> Keras (using model parameters)
    Confirmed the same accuracy as "predict_chainer"

    The loss function in training at chainer is "softmax_cross_entropy"
    It was reproduced by using "softmax" for the output layer and
    "categorical_crossentropy" as the loss function.

    model compile parameters:
     * loss='categorical_crossentropy'
     * optimizer=Adam
     * metrics=['accuracy']
    :param param_path:
    :return:
    """
    # Model structure
    input_shape = (28 * 28,)
    # Layers
    layers = [Dense, Dense, Dense]
    # units
    units = [1000, 1000, 10]
    # layer activations
    activations = ['relu', 'relu', 'softmax']

    # Load chainer model
    print('Load File (%s)' % param_path)
    try:
        w1, w2, w3, b1, b2, b3 = _load_param(param_path)

        # weight, bias
        params = [[w1, b1], [w2, b2], [w3, b3]]

    except Exception as e:
        sys.stderr.write(str(e))
        return None

    # Restore to the model of Keras
    try:
        model = _create_model(input_shape=input_shape,
                              layers=layers, units=units,
                              params=params, activations=activations)

        # model compile parameters
        loss_func = 'categorical_crossentropy'
        optimizer = 'adam'
        metrics = ['accuracy']
        # compile
        model.compile(loss=loss_func, optimizer=optimizer, metrics=metrics)

    except Exception as e:
        sys.stderr.write(str(e))
        return None

    print('Compiled model')
    model.summary()

    return model


def _create_model(input_shape, layers, units, params, activations):
    model = Sequential()
    # Dense:
    # kernel_initializer='glorot_uniform'
    # (chainer: i.i.d. Gaussian samples)
    # bias_initializer='zeros'

    # input layer
    model.add(
        layers[0](units=units[0], activation=activations[0], weights=params[0],
                  input_shape=input_shape, use_bias=True))

    for layer, unit, activation, param in zip(layers[1:], units[1:],
                                              activations[1:], params[1:]):
        # add layer
        model.add(
            layer(units=unit, activation=activation, weights=param,
                  use_bias=True))

    return model


def _load_param(param_path):
    """
    Loaad the weight and bias of each layer
    :param param_path:
    :return: The weight and bias of the three layers
    """
    # load npz
    param = np.load(param_path)
    name_list = param['arr_0']
    param_list = param['arr_1']

    w1 = w2 = w3 = b1 = b2 = b3 = []

    # Premise: The number of name_list is the same as the number of param_list
    for name, param in zip(name_list, param_list):

        if 'W' in name:
            # set weight
            if '1' in name:
                w1 = param.T

            elif '2' in name:
                w2 = param.T

            elif '3' in name:
                w3 = param.T

        elif 'b' in name:
            # set bias
            if '1' in name:
                b1 = param

            elif '2' in name:
                b2 = param

            elif '3' in name:
                b3 = param

    return w1, w2, w3, b1, b2, b3


def _print_model_info(model):

    print('')
    sep = '_________________________________________________________________'

    iter_weights = iter(model.get_weights())
    for index, (weight, bias) in enumerate(zip(iter_weights, iter_weights)):
        print(sep)

        print('Layer :', index)
        print('Weight :', np.shape(weight))
        print(weight)
        print('Bias :', np.shape(bias))
        print(bias)

        print(sep)


def _evaluate_model(model, x, y):
    loss, accuracy = model.evaluate(x, y, verbose=1)
    print('')
    print('loss :', loss)
    print('accuracy :', accuracy)


def _check_param(model):
    # debug
    print('check load param:')
    p_path = pathlib.Path(load_dir, 'mnist_chainer/chainer_model_mnist.npz')
    w1, w2, w3, b1, b2, b3 = _load_param(str(p_path))

    for i, (v1, v2) in enumerate(
            zip([w1, b1, w2, b2, w3, b3], model.get_weights())):

        if i % 2 is 0:
            pf = 'w'
        else:
            pf = 'b'

        flag = np.allclose(v1, v2)

        print('%s_%d :' % (pf, i // 2), flag)
        if not flag:
            print(v1 - v2)


if __name__ == '__main__':
    load_dir = '/home/sato/Documents/OSARA/downloaded_data'
    k_model = model_load(load_dir)

    # evaluate
    from keras.datasets import mnist
    import keras.utils

    # Load Test Data
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_test = X_test.reshape(10000, 28 * 28).astype('float32') / 255
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # train = True
    train = False

    if train:
        # Save load model
        k_model.save('mnist_mlp_load_chainer_model.h5')

        # training load model
        X_train = X_train.reshape(60000, 28 * 28).astype('float32') / 255
        Y_train = keras.utils.to_categorical(Y_train, 10)

        batch_size = 100
        epochs = 20
        verbose = 2

        k_model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs, verbose=verbose)

        # save keras new model
        k_model.save('mnist_mlp_train_model.h5')

    else:
        # Read the parameters of the model created with chainer
        print('----------param_load_model----------')
        _print_model_info(k_model)
        _evaluate_model(k_model, X_test, Y_test)
        _check_param(k_model)

        # Read the parameters of the model created with chainer and
        # load the model save by keras
        print('----------keras_save_model----------')
        keras_save_model = keras.models.load_model(
            'mnist_mlp_load_chainer_model.h5')
        _print_model_info(keras_save_model)
        _evaluate_model(keras_save_model, X_test, Y_test)
        _check_param(keras_save_model)

        # Read the parameters of the model created with chainer,
        # train by keras and load the saved model
        print('----------new_train_model----------')
        new_train_model = keras.models.load_model(
            'mnist_mlp_train_model.h5')
        _print_model_info(new_train_model)
        _evaluate_model(new_train_model, X_test, Y_test)
        _check_param(new_train_model)
