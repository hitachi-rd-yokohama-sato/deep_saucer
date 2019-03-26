# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from lib.demo_template import DemoTemplate
from lib.demo_utils import load_config, load_module
from lib.converter import ArgumentGenerator
from lib.utils.utils import read_csv


def get_random_data(data, n=10, seed=1):
    """
    :param data: np.ndarray
    :param n: int
    :param seed: int
    :return: np.ndarray
    """
    np.random.seed(seed)
    ind = np.random.choice(data.shape[0], n)
    return data[ind]


def main(model, dataset, config_path):
    config = load_config(config_path)
    reference_model_path = config["rule"]
    evaluation_criteria_path = config["rdg_weight"]
    arguments_path = config["rdg_output"]
    option_function_paths = config["rdg_exe"]
    function_names = config["function_names"]

    args = read_csv(arguments_path)
    modules = load_module(option_function_paths, function_names)
    print('Convert Test Input Data')
    convert_datasets = [module(dataset) for module in modules]
    arg_gen = ArgumentGenerator(args, convert_datasets)
    demo = DemoTemplate(model=model, dataset=dataset,
                        reference_model=reference_model_path,
                        evaluation_criteria=evaluation_criteria_path,
                        arguments=arg_gen,
                        plot=False, debug=False)
    demo.play()
