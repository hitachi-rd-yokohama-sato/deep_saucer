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
# DataSet loading script for DNN Coverage used with DeepSaucer

## Requirement
Same as DNN Coverage project

## Directory Structure

Any Directory
`-- DeepSaucer
    `-- mnist
        `-- data
            |-- dataset_neuron_coverage_keras.py @
            |-- dataset.py
            |-- dataset_test.py
            `-- dataset_test_images.py (call data_create)
"""

from keras import backend as K
import dataset_test_images


# input image dimensions
img_rows, img_cols = 28, 28


def data_create(downloaded_data):
    x_test = dataset_test_images.data_create(downloaded_data)

    print('keras image_data_format: {}'.format(K.image_data_format()))
    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    return x_test
