#!/bin/bash
#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
pip install --upgrade pip
conda create -n dnn_encoding python=3.6 -y
source activate dnn_encoding
pip install tensorflow==1.12.0
pip install z3-solver
pip install pyparsing
