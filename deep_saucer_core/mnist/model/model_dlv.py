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
# Model loading script for DLV used with DeepSaucer

## Requirement
Same as DLV project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- mnist
|       `-- model
|           `-- model_dlv.py @
`-- dlv (_dlv_dir)
    `-- networks (_dlv_networks_dir)
        |-- mnist (_mnist_dir)
        |   |-- mnist.mat (mat_file)
        |   `-- mnist.json (model_file)
        `-- mnist_network.py (call read_model_from_file)
"""

import os
import sys

_root_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_dlv_dir = os.path.join(_root_dir, 'dlv')
_dlv_networks_dir = os.path.join(_dlv_dir, 'networks')
_mnist_dir = os.path.join(_dlv_networks_dir, 'mnist')

sys.path.append(_dlv_dir)
sys.path.append(_dlv_networks_dir)

from mnist_network import read_model_from_file


def model_load(downloaded_data):
    """
    Load mnist model of DLV project
    :return:
    """
    mat_file = os.path.join(_mnist_dir, 'mnist.mat')
    model_file = os.path.join(_mnist_dir, 'mnist.json')

    if os.path.isfile(mat_file) and os.path.isfile(model_file):
        model = read_model_from_file(mat_file, model_file)
    else:
        return None

    return model
