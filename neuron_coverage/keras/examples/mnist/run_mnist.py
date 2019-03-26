# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import load_model, Sequential
from keras.datasets import mnist
from keras import backend as K

import os
import numpy as np
import sys
from pathlib import Path

_proj_dir = Path(__file__).absolute().parent.parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_mnist = Path(_proj_dir, 'examples', 'mnist')

sys.path.append(str(_proj_dir))
sys.path.append(str(_lib_dir))
from lib.coverage_verification import main

_model_name = 'mnist_cnn.h5'
_conf_name = 'config.json'


def get_mnist_model(model_path, img_rows=28, img_cols=28,
                    batch_size=128, num_classes=10, epochs=12):
    if os.path.isfile(model_path):
        print('Load Model : %s' % model_path)
        return load_model(model_path)

    else:
        print('%s is not found.' % model_path)
        print('Start of create and training to model')
        return train_mnist(model_path,
                           img_rows, img_cols, batch_size, num_classes, epochs)


def train_mnist(model_path,
                img_rows, img_cols, batch_size, num_classes, epochs):
    # the model, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(model_path)

    return model


def get_mnist_dataset(img_rows=28, img_cols=28):
    _, (x_test, _) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    return x_test


if __name__ == '__main__':
    # Model file path to use
    mnist_model_path = Path(_examples_mnist, 'model', _model_name)

    # Load model
    mnist_model = get_mnist_model(str(mnist_model_path))

    # Load Dataset
    test_x = np.array(get_mnist_dataset())

    # Config Path
    conf_dir = Path(_lib_dir, 'config')
    conf_path = Path(conf_dir, _conf_name)

    # Run DNN Coverage
    main(mnist_model, test_x, str(conf_path))
