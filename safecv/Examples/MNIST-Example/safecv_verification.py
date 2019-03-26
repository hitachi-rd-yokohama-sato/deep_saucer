# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 :
# Derived from MNIST-SafeCV.py
#******************************************************************************************
import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# SafeCV/Examples/MNIST-Example/
_mnist_example_dir = os.path.dirname(os.path.abspath(__file__))
# SafeCV/Examples/
_examples_dir = os.path.dirname(_mnist_example_dir)
# SafeCV/
_safecv_dir = os.path.dirname(_examples_dir)
# Append SafeCV/SafeCV
sys.path.append(os.path.join(_safecv_dir, "SafeCV"))
from MCTS import MCTS, MCTS_Parameters


def max_manip(p,e):
    w = 255-p
    if(p < w):
        return 255
    else:
        return 0


def main(model, dataset=None, config_path=None):
    """
    SafeCV/Example/MNIST-Example/MNIST-SafeCV.py(custom)
    Do not install SafeCV library, use customized MCST.py and DFMCS.py
    :param dataset: test_images, test_labels (tuple)
    :param model:
    :param config_path: JSON file path that set the index to be tested
    :return:
    """
    x_test, y_test = dataset

    # get index
    j_data = json.load(open(config_path))
    imnum = j_data['imnum']


    params_for_run = MCTS_Parameters(x_test[imnum], y_test[imnum], model, predshape=(1,1,28,28))
    params_for_run.X_SHAPE = 28
    params_for_run.Y_SHAPE = 28
    params_for_run.small_image = True
    params_for_run.verbose = True
    params_for_run.MANIP = max_manip
    params_for_run.VISIT_CONSTANT = 1
    params_for_run.VISIT_CONSTANT = 6
    params_for_run.simulations_cutoff = 100
    params_for_run.backtracking_constant = 10

    best_image, sev, prob, statistics = MCTS(params_for_run)


    # In[3]:


    print("BEST ADVERSARIAL EXAMPLE:")
    plt.imshow(best_image)
    plt.show()
    prob = model.predict(best_image.reshape(1,1,28,28))
    new_class = np.argmax(prob[0])
    new_prob = prob[0][np.argmax(prob)]
    print("True class: %s; Predicted as: %s with confidence: %s; After %s manipulations"%(y_test[imnum], new_class, new_prob, sev ))
    plt.clf()
    print("MCTS Run analysis:")
    a, = plt.plot(statistics[0], label="Min Severity Found")
    b, = plt.plot(statistics[1], label="Severity per Iteration")
    c, = plt.plot(statistics[2], label="Rolling Average Severity")
    plt.legend(handles=[a,b,c], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title("Single Run MCTS Statisitcs")
    plt.xlabel("Iteration")
    plt.ylabel("L_0 Severity")
    plt.show()
