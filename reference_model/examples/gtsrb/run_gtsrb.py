# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import sys
from pathlib import Path

_proj_dir = Path(__file__).absolute().parent.parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_gtsrb = Path(_proj_dir, 'examples', 'gtsrb')

sys.path.append(str(_proj_dir))
sys.path.append(str(_lib_dir))
from lib.ref_model_verification import main
from lib.dataset import data_create
from lib.model import model_load


_config_name = 'config_gtsrb.json'

if __name__ == '__main__':

    # Model file path to use
    gtsrb_model_dir = Path(_examples_gtsrb, 'model')

    # Dataset file path to use
    gtsrb_dataset_dir = Path(_examples_gtsrb, 'data')

    # Load model
    model = model_load(str(gtsrb_model_dir))

    # Load dataset
    dataset = data_create(str(gtsrb_dataset_dir))

    # Config file path to use
    json_path = Path(_lib_dir, 'configs', _config_name)

    # Run Reference Model
    main(model, dataset, str(json_path))
