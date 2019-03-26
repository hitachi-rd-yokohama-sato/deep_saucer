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
# DataSet loading script used with DeepSaucer

## Directory Structure

Any Directory
`-- DeepSaucer
    |-- downloaded_data (downloaded_path)
    |   `-- mnist_chainer (dl_dir)
    |       |-- t10k-images-idx3-ubyte.gz
    |       |-- t10k-labels-idx1-ubyte.gz
    |       |-- train-images-idx3-ubyte.gz
    |       `-- train-labels-idx1-ubyte.gz
    `-- mnist
        `-- data
            `-- dataset_c2k.py @
"""

import numpy as np
import pathlib
from tensorflow.examples.tutorials.mnist import input_data


def data_create(downloaded_path):
    dl_dir = pathlib.Path(downloaded_path, 'mnist_chainer')
    #save_dir = dl_dir.joinpath('input_data')
    save_dir = dl_dir

    print('Refer Directory (%s)' % save_dir)
    mnist_dataset = input_data.read_data_sets(str(save_dir), one_hot=True)

    print('shape :', np.shape(mnist_dataset.test.images))

    return np.array(mnist_dataset.test.images)
