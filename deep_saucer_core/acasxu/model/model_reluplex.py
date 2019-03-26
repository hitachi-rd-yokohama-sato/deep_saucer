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
# Model loading script for Reluplex used with DeepSaucer

## Requirement
Same as Reluplex project and Python 3.6

## Directory Structure

Any Directory (_root_dir)
|-- DeepSaucer
|   `-- acasxu
|       `-- model_reluplex.py @
`-- ReluplexCav2017 (_reluplex_dir)
    `-- nnet (_nnet_dir)
        `-- ACASXU_run2a_1_1_batch_2000.nnet (model_path)
"""
import os
import urllib.request

from pathlib import Path
from getpass import getpass


url_base = r'https://raw.githubusercontent.com/guykatzz/ReluplexCav2017/master/nnet/'
file_name = 'ACASXU_run2a_1_1_batch_2000.nnet'

_root_dir = Path(__file__).parent.parent.parent.parent
_reluplex_dir = Path(_root_dir, 'relplex')
_nnet_dir = Path(_reluplex_dir, 'nnet').absolute()


def model_load(downloaded_path):

    # Save Path
    model_path = Path(_nnet_dir, file_name)
    if model_path.exists():
        print('Use Local Model (%s)' % str(model_path.absolute()))
        return str(model_path.absolute())

    # DownLoad
    print('Download Model (%s)' % url_base + file_name)
    urllib.request.urlretrieve(url_base + file_name, model_path.absolute())

    return str(model_path.absolute())
