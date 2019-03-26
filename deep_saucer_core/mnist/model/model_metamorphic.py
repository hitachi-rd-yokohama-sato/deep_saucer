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
# Model loading script for Metamorphic Test used with DeepSaucer

## Requirement
Same as Metamorphic Test project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- mnist
|       `-- model
|           `-- model_metamorphic.py @
`-- metamorphic_testing (_metamorphic_dir)
    `-- lib (_metamorphic_lib)
        `-- utils (_metamorphic_utils)
            `-- structutil.py (use NetworkStruct)
"""
from pathlib import Path

import tensorflow as tf


def model_load(downloaded_data):
    model_path = Path(downloaded_data, 'mnist_tensorflow_metamorphic')
    ckpt_path = model_path.joinpath('model_mnist.ckpt')
    meta_path = Path(str(ckpt_path) + '.meta')

    sess = tf.Session()
    saver = tf.train.import_meta_graph(str(meta_path))
    saver.restore(sess, str(ckpt_path))
    return sess
