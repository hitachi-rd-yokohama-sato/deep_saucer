# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import numpy as np
from keras.models import load_model

# model_path = "option_functions/shape_0.9934.h5"
from pathlib import Path
model_path = str(Path(Path(__file__, ).parent, "shape_0.9934.h5").absolute())


def get_predict_result(inputs, model):
    # check function
    return np.argmax(model.predict(inputs), axis=1)


def get_shape(inputs):
    model = load_model(model_path)
    return get_predict_result(inputs, model)
