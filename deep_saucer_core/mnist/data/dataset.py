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
# MNIST Dataset loading script used with DeepSaucer

## Directory Structure

Any Directory
`-- DeepSaucer
    `-- mnist
        `-- data
            `-- dataset.py @
"""

from keras.datasets import mnist


def data_create(downloaded_data):
    """
    Load mnist dataset from keras library
    :param downloaded_data: 
    :return: 
    """
    return mnist.load_data()
