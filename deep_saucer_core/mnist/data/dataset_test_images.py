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
# MNIST DataSet loading script used with DeepSaucer

## Directory Structure

Any Directory
`-- DeepSaucer
    `-- mnist
        `-- data
            |-- dataset.py
            |-- dataset_test.py (call data_create)
            `-- dataset_test_images.py @
"""
import dataset_test


def data_create(downloaded_data):
    """
    Load mnist test images from keras library
    :param downloaded_data: 
    :return: test images
    """
    x_test, _ = dataset_test.data_create(downloaded_data)
    return x_test
