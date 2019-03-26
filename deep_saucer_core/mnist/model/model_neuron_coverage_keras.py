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
# Model loading script for DNN Coverage used with DeepSaucer

## Requirement
Same as DNN Coverage project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- mnist
|       `-- model
|           `-- model_neuron_coverage_keras.py @
`-- neuron_coverage
    `-- keras (_dnn_coverage_dir)
        `-- examples
            `-- mnist (_example_mnist_dir)
                |-- run_mnist.py (call get_mnist_model)
                `-- model (_model_dir)
                    `--- mnist_cnn.h5 (_model_path)
"""

import sys
from pathlib import Path

_root_dir = Path(__file__).absolute().parent.parent.parent.parent
_dnn_coverage_dir = Path(_root_dir, 'neuron_coverage', 'keras')
_examples_mnist_dir = Path(_dnn_coverage_dir, 'examples', 'mnist')

_file_name = 'mnist_cnn.h5'
_model_dir = Path(_examples_mnist_dir, 'model')
_model_path = Path(_model_dir, _file_name)

sys.path.append(str(_dnn_coverage_dir))
sys.path.append(str(_examples_mnist_dir))
from run_mnist import get_mnist_model


def model_load(downloaded_data):
    """
    Load mnist model of DNN Coverage project
    :return:
    """
    return get_mnist_model(str(_model_path))
