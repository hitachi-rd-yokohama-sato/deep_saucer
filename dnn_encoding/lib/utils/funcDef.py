#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from z3 import *
from math import exp


def Relu(x):
    return If(x >= 0, x, 0)


def Sigmoid(x):
    # return 1 / (1 + exp(1) ** (-x))
    return If(x >= 0, 1, 0)


def calcExp(x):
    return exp(1) ** x
