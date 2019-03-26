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
# Model loading script for Reference Model used with DeepSaucer

## Requirement
Same as Reference Model project

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   |-- downloaded_data (downloaded_path)
|   |   `-- vggnet_model_acc_0.69.hdf5
|   `-- gtsrb
|       `-- model
|           `-- model_ref_model.py @
`-- reference_model (_ref_model_dir)
    `-- lib
        `-- model.py (call model_load)
"""
import sys
from pathlib import Path

_root_dir = Path(__file__).absolute().parent.parent.parent.parent
# HACK new_project
_ref_model_dir = Path(_root_dir, 'reference_model')
sys.path.append(str(_ref_model_dir))
import model


def model_load(downloaded_path):
    # Load "vggnet_model_acc_0.69.hdf5"
    return model.model_load(downloaded_path)
