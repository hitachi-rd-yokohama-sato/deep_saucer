# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from keras.models import load_model as keras_load_model
from pathlib import Path


def model_load(root_path):
    """
    Notice that it is a calling function dedicated to OSARA Tool.

    Load model(keras.models.Sequential format).

    !Required: Function Name is "model_load"

    :type root_path: str
    :rtype: keras.models.Sequential

    >>> root_path = r"C:\cygwin64\home\spc000\python\\new_project\model"
    >>> model = model_load(root_path)
    """
    root_path = Path(root_path)
    model_path = root_path.joinpath("vggnet_model_acc_0.69.hdf5")
    model = keras_load_model(model_path)
    return model
