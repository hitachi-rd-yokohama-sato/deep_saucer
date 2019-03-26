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
            |-- dataset_neuron_coverage_tensorflow_native.py @
            |-- dataset.py
            `-- dataset_test.py (call data_create)
"""
from pathlib import Path
from tensorflow.examples.tutorials.mnist import input_data


# input image dimensions
img_rows, img_cols = 28, 28


def data_create(downloaded_data):
    data_path = Path(downloaded_data).absolute().joinpath(
        'mnist_tensorflow_neuron_coverage')
    dataset = input_data.read_data_sets(str(data_path), one_hot=True)

    return [dataset.test.images, dataset.test.labels, 1.0]
