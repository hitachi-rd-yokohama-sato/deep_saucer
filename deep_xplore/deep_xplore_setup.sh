#!/bin/bash
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************

pip install --upgrade pip
conda create -n deep_xplore python=2.7.12 -y
source activate deep_xplore
pip install keras==2.0.8
pip install tensorflow==1.3.0
pip install pillow
pip install h5py
pip install opencv-python
