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
# Model loading script for DeepXplore used with DeepSaucer

## Requirement
Same as DeepXplore project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- mnist
|       `-- model
|           `-- model_deepxplore.py @
`-- deep_xplore (_deepxplore_dir)
    `-- MNIST (_deepxplore_mnist_dir)
        |-- Model1.py (call Model1)
        |-- Model1.h5
        |-- Model2.py (call Model2)
        |-- Model2.h5
        |-- Model3.py (call Model3)
        `-- Model3.h5
"""

from keras.layers import Input
import sys
import os

_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_deepxplore_dir = os.path.join(_root_dir, 'deep_xplore')
_deepxplore_mnist_dir = os.path.join(_deepxplore_dir, 'MNIST')

sys.path.append(_deepxplore_dir)
sys.path.append(_deepxplore_mnist_dir)

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3


def model_load(downloaded_path):
    """
    Load list of three models and keras.layers.Input()
    :param downloaded_path:
    :return: List containing three models and keras.layers.Input()
    """
    input_shape = (28, 28, 1)

    # define input tensor as a placeholder
    input_tensor = Input(shape=input_shape)

    # load multiple models sharing same input tensor
    train = not os.path.isfile(os.path.join(_deepxplore_mnist_dir, 'Model1.h5'))
    model1 = Model1(input_tensor=input_tensor, train=train)

    train = not os.path.isfile(os.path.join(_deepxplore_mnist_dir, 'Model2.h5'))
    model2 = Model2(input_tensor=input_tensor, train=train)

    train = not os.path.isfile(os.path.join(_deepxplore_mnist_dir, 'Model3.h5'))
    model3 = Model3(input_tensor=input_tensor, train=train)

    return model1, model2, model3, input_tensor
