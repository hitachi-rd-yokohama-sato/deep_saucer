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
from pathlib import Path

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def data_create(downloaded_data):
    data_path = Path(downloaded_data).absolute().joinpath('mnist_tensorflow_metamorphic')
    dataset = input_data.read_data_sets(str(data_path), one_hot=True)
    
    return [dataset.test.images]
