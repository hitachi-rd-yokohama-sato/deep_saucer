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

Any Directory (_root_dir)
|-- DeepSaucer
|   |-- downloaded_data(downloaded_path)
|   |   `-- test_dataset_3_32_32.h5 (dl_data_path)
|   `-- gtsrb
|       `-- data
|           `-- dataset_neuron_coverage_keras.py @
`-- neuron_coverage
    `-- keras (_dnn_coverage_dir)
        `-- examples
            `-- gtsrb (_example_gtsrb_dir)
                `-- run_gtsrb.py (call get_gtsrb_dataset)
"""

import sys
from pathlib import Path

_root_dir = Path(__file__).absolute().parent.parent.parent.parent
_dnn_coverage_dir = Path(_root_dir, 'neuron_coverage', 'keras')
_examples_gtsrb_dir = Path(_dnn_coverage_dir, 'examples', 'gtsrb')

_file_name = 'test_dataset_3_32_32.h5'

sys.path.append(str(_dnn_coverage_dir))
sys.path.append(str(_examples_gtsrb_dir))
from run_gtsrb import get_gtsrb_dataset


def data_create(downloaded_path):
    """
    Load GTSRB dataset of DNN_Coverage project
    :return:
    """
    dl_data_path = Path(downloaded_path, _file_name)
    return get_gtsrb_dataset(str(dl_data_path))
