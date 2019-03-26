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
|           `-- model_neuron_coverage_tensorflow_native.py @
`-- metamorphic_testing (_metamorphic_dir)
    `-- lib (_metamorphic_lib)
        |-- examples
        |-- tf_ckpt
        |   | model.ckpt.data-00000-of-00001
        |   | model.ckpt.index
        |   | model.ckpt.meta
        |   ` model.ckpt_name.json
        `-- utils (_metamorphic_utils)
            `-- structutil.py (use NetworkStruct)
"""
from pathlib import Path

import tensorflow as tf

_root_dir = Path(__file__).absolute().parent.parent.parent.parent
_dnn_coverage_dir = Path(_root_dir, 'neuron_coverage', 'tensorflow_native')
_examples_ckpt_dir = Path(_dnn_coverage_dir, 'lib', 'examples', 'tf_ckpt')


def model_load(downloaded_data):
    model_path = _examples_ckpt_dir
    ckpt_path = model_path.joinpath('model.ckpt')
    meta_path = Path(str(ckpt_path) + '.meta')

    sess = tf.Session()
    saver = tf.train.import_meta_graph(str(meta_path))
    saver.restore(sess, str(ckpt_path))
    return sess
