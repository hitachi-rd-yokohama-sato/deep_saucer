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
conda create -n safecv python=2.7 -y
source activate safecv
#pip install SafeCV
pip install keras==1.2.2
pip install tensorflow
pip install matplotlib
pip install pomegranate
pip install opencv-python==3.3.1.11
pip install opencv-contrib-python==3.3.1.11

