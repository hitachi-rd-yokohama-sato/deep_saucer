# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from keras.models import load_model
from keras.utils import HDF5Matrix
import numpy as np
import sys

from pathlib import Path

_proj_dir = Path(__file__).absolute().parent.parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_gtsrb = Path(_proj_dir, 'examples', 'gtsrb')

sys.path.append(str(_proj_dir))
sys.path.append(str(_lib_dir))
from lib.coverage_verification import main


_model_name = 'gtrsb_image_classification_model.hdf5'
_dataset_name = 'test_dataset_3_32_32.h5'
_conf_name = 'config.json'


def get_gtsrb_model(model_path):
    print('Load Model : %s' % model_path)
    return load_model(model_path)


def get_gtsrb_dataset(dataset_path):
    print('Load DataSet : %s' % dataset_path)
    x_test = HDF5Matrix(dataset_path, "images")
    x_test = np.array(x_test)

    return x_test


if __name__ == '__main__':
    # Model file path to use
    gtsrb_model_path = Path(_examples_gtsrb, 'model', _model_name)

    # Dataset file path to use
    gtsrb_dataset_path = Path(_examples_gtsrb, 'data', _dataset_name)

    # Load model
    model = get_gtsrb_model(str(gtsrb_model_path))

    # Load Dataset
    test_x = get_gtsrb_dataset(str(gtsrb_dataset_path))

    # Config Path
    conf_dir = Path(_lib_dir, 'config')
    conf_path = Path(conf_dir, _conf_name)

    # Run DNN Coverage
    main(model, test_x, str(conf_path))
