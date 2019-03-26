# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import json
import sys
import webbrowser
from pprint import pprint

import pandas as pd
from keras import backend as k
# Support Layers
from keras.layers import (
    Dense, Activation, Conv1D, Conv2D, SeparableConv2D, Conv2DTranspose, Conv3D,
    LocallyConnected1D, LocallyConnected2D, SimpleRNN, GRU, LSTM, ConvLSTM2D)

DETERMINATION_ON_ACTIVATION = 'determination_on_activation'
THRESHOLD = 'threshold'
LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'
ACTIVATION_FILTER_NO = 'activation_filter_no'
REPLACE_NUM = 'replace_num'
ACTIVATION_VALUE = 'activation_value'
ACTIVATION_VALUE_ADD_NEURON_VARIATION = 'heat_map_type'
# SPLIT_DATASET_START = 'split_dataset_start'
# SPLIT_DATASET_END = 'split_dataset_end'
# COV_FIRST = 'cov_first'
# COV_SECOND = 'cov_second'

import sys
from pathlib import Path

_dnn_coverage_dir = Path(__file__).absolute().parent.parent
_lib_dir = Path(_dnn_coverage_dir, 'lib')
sys.path.append(str(_dnn_coverage_dir))
sys.path.append(str(_lib_dir))

from nnutil import Network, get_outputs_per_layer, Coverage, \
    Density, density_to_heatmap_html, Graph, get_outputs_per_input, \
    is_target_layer


# def density_to_csv(data, file_name):
#     dfs = []
#     for layer_no, layer in enumerate(data, 1):
#         d = pd.DataFrame({"layer": [layer_no for _ in range(layer.size)],
#                           "unit": [u for u in range(layer.size)],
#                           "value": [unit for unit in layer[0]]})
#         d["value"] = d["value"].round(10)
#         dfs.append(d)
#     master_df = pd.concat(dfs, ignore_index=True)
#     # file_name = "vggnet_input_{}.csv".format(data_length)
#     pd.DataFrame(master_df).to_csv(file_name, index=False)


def apply_argument(json_data, dataset):
    # Method of argument error handling
    def argument_exception_trow(error_message):
        raise ValueError("Argument exception：" + error_message)
        # try:
        #     raise ValueError("Argument exception：" + error_message)
        # except ValueError as e:
        #     sys.stderr.write(str(e))
        #     sys.exit()

    x_input = dataset

    # TODO Split Dataset
    # # Setting the number of input data
    # split_dataset_start = json_data[SPLIT_DATASET_START]
    # split_dataset_end = json_data[SPLIT_DATASET_END]
    #
    # # Argument check
    # if type(split_dataset_end) is int or type(split_dataset_start) is int:
    #     if split_dataset_end < 1 or split_dataset_start < 0:
    #         argument_exception_trow(
    #             "The value of split_dataset_(start or end) must be "
    #             "an integer greater than or equal to 1.")
    #     elif split_dataset_start > split_dataset_end:
    #         argument_exception_trow(
    #             "The value of split_dataset_start must be "
    #             "an integer greater than split_dataset_end.")
    # else:
    #     argument_exception_trow(
    #         "The value of split_dataset_(start and end) must be "
    #         "an integer greater than or equal to 1.")
    #
    # # Dataset split
    # x_input = dataset[split_dataset_start:split_dataset_end]

    # Method for determining activation
    determination_on_activation = json_data[DETERMINATION_ON_ACTIVATION]

    # A method for determining an activation value to be added to a neuron
    activation_value_add_neuron_variation = 1
    if ACTIVATION_VALUE_ADD_NEURON_VARIATION in json_data:
        activation_value_add_neuron_variation = json_data[ACTIVATION_VALUE_ADD_NEURON_VARIATION]

    # default value
    threshold, lower_bound, activation_filter_no = 0.0, 0.0, 1

    # Method for determining activation
    if determination_on_activation == 0:
        # 0:Threshold or more active
        # Threshold (threshold)
        if THRESHOLD in json_data:
            threshold = json_data[THRESHOLD]

        if threshold == "":
            # Setting default values
            threshold = 0.0

        lower_bound = ""
        upper_bound = ""
        activation_filter_no = ""

    elif determination_on_activation == 1:
        # 1:Activate within lower_bound or more upper_bound
        # Lower limit (lower_bound)
        if LOWER_BOUND in json_data:
            lower_bound = json_data[LOWER_BOUND]

        # Upper limit (upper_bound)
        upper_bound = ""
        if UPPER_BOUND in json_data:
            upper_bound = json_data[UPPER_BOUND]

        threshold = ''
        activation_filter_no = ""

        # Argument check
        if type(lower_bound) is str:
            argument_exception_trow(
                "The value of lower_bound is not set. "
                "Or it is not a numeric type.")
        if type(upper_bound) is str:
            argument_exception_trow(
                "The value of upper_bound is not set. "
                "Or it is not a numeric type.")
        if lower_bound > upper_bound:
            argument_exception_trow(
                "The value of upper_bound must be greater than lower_bound.")

    elif determination_on_activation == 2:
        # 2: Activate up to μ μth layer in each layer
        # Not use
        lower_bound = ""
        upper_bound = ""
        threshold = ''

        # Upper μ-number(activation_filter_no)
        if ACTIVATION_FILTER_NO in json_data:
            activation_filter_no = json_data[ACTIVATION_FILTER_NO]

        if type(activation_filter_no) is int:
            if activation_filter_no < 1:
                argument_exception_trow(
                    "The value of activation_filter_no must be "
                    "an integer greater than or equal to 1.")
        else:
            argument_exception_trow(
                "The value of activation_filter_no must be "
                "an integer greater than or equal to 1.")
    else:
        argument_exception_trow(
            "The value of determination_on_activation must be 0 or 1 or 2.")

    # A method for determining an activation value to be added to a neuron
    # A：Simple increment
    if activation_value_add_neuron_variation in [0, 1, 2]:
        # 1：Activity should be 1. The inactivity is set to 0
        replace_num = 1
        # 0：Add the value of activated neurons
        activation_value = 0

    # B：Concentration coverage
    elif activation_value_add_neuron_variation == 3:
        # 1：Activity should be 1. The inactivity is set to 0
        replace_num = 1
        # 1：We add the value that divides the value of activated neurons by weight
        activation_value = 1

    # TODO Concentration based on activity value
    # # C1：Concentration based on activity value
    # elif activation_value_add_neuron_variation == 4:
    #     # 0：Activity is a value through activation function, inactivity is set to 0
    #     replace_num = 0
    #     # 0：Add the value of activated neurons
    #     activation_value = 0
    #
    # # C2：Concentration based on activity value
    # elif activation_value_add_neuron_variation == 5:
    #     # 0：Activity is a value through activation function, inactivity is set to 0
    #     replace_num = 0
    #     # 1：We add the value that divides the value of activated neurons by weight
    #     activation_value = 1
    else:
        argument_exception_trow(
            "The value of activation_value_add_neuron_variation must be 0 or 1 or 2 or 3.")

    # Generate arguments
    determination_on_activation_kw = {
        DETERMINATION_ON_ACTIVATION: determination_on_activation,
        THRESHOLD: threshold,
        LOWER_BOUND: lower_bound,
        UPPER_BOUND: upper_bound,
        ACTIVATION_FILTER_NO: activation_filter_no}
    activation_value_add_neuron_kw = {
        ACTIVATION_VALUE_ADD_NEURON_VARIATION: activation_value_add_neuron_variation,
        REPLACE_NUM: replace_num,
        ACTIVATION_VALUE: activation_value}

    return determination_on_activation_kw, activation_value_add_neuron_kw, \
        x_input


def check_layer_filters(filters_name):
    """
    Check layer names and retern filter layer class
    :param filters_name: Layer names
    :return: Layer Class list
    """
    # Get not import layers
    non_target_layers = [fn for fn in filters_name
                         if fn not in globals().keys()]

    for ln in non_target_layers:
        if ln not in globals().keys():
            print('[Warning] "%s" is non-targeted layer' % ln)

    return [globals()[n] for n in filters_name if n not in non_target_layers]


def judge_layers(model, filters):
    """
    Judge layer in model
    :param model:
    :param filters: filter layers
    :return:
    """
    judged_layers = set()
    for layer in model.layers:
        if type(layer) in judged_layers:
            continue
        if not is_target_layer(layer, filters):
            print('[Warning] "%s" is non-targeted layer'
                  % layer.__class__.__name__)
        else:
            print('"%s" is targeted layer'
                  % layer.__class__.__name__)

        # Add appeared layer
        judged_layers.add(type(layer))


def main(model, dataset=None, config_path=None):
    """
    Measure the coverage of the leyer with "Activation" as an attribute.
    :param model: test model
    :param dataset: test dataset
    :param config_path: config
    :return:
    """
    try:
        # Check and apply argument
        with open(config_path) as fs:
            json_data = json.load(fs)
        determination_on_activation_kw, activation_value_add_neuron_kw, \
            x_input = apply_argument(json_data, dataset)

        print('----------------------------------------')
        print('Dataset Shepe:')
        pprint(x_input.shape)
        print('----------------------------------------')

        # The layer not to have activation attribute becomes the error
        filters_name = ['Dense', 'Conv2D']

        # Check Layer Name
        layer_filters = check_layer_filters(filters_name)

        if len(layer_filters) > 0:
            print('Use Layer Filter: [%s]' %
                  ', '.join([l.__name__ for l in layer_filters]))
        else:
            print('Not use Layer Filter')

        print('Layers:')
        # Warning to non-targeted layers
        judge_layers(model, layer_filters)

        print('----------------------------------------')

        print('Coverage measurement in progress...')
        inp = get_outputs_per_layer(model, layer_filters, x_input)
        network = Network(model, layer_filters, inp,
                          determination_on_activation_kw,
                          activation_value_add_neuron_kw)

        coverage = Coverage()

        # Output the coverage ratio of all layer
        print('Coverage rate all layer:  ' + str(
            coverage.activation_rate(network)))
        # Output the coverage ratio of each layer
        layer_name_list = []
        for i in network:
            print('Coverage rate one layer ' + str(i) + ':  ' + str(
                coverage.activation_rate(i)))
            layer_name_list.append(str(i))

        # network, inp, coverage = '', '', ''

        activation_value_add_neuron_variation = activation_value_add_neuron_kw[ACTIVATION_VALUE_ADD_NEURON_VARIATION]
        activation_value = activation_value_add_neuron_kw[ACTIVATION_VALUE]
        if activation_value_add_neuron_variation is not 0:
            # Generate html of heat map image
            print('----------------------------------------')
            print('Analyzing Model for creating Heat Map...')
            inp_graph = get_outputs_per_input(model, layer_filters, x_input)
            graph = Graph(model, layer_filters, inp_graph,
                          determination_on_activation_kw,
                          activation_value_add_neuron_kw)

            density = Density(graph)
            density.update(activation_value)
            html_file_pass = density_to_heatmap_html(density, activation_value_add_neuron_variation)
            print('Output Heat Map: %s' % html_file_pass)
            # inp_graph, graph = '', ''

            # TODO Multiple layers combination coverage rate
            # try:
            #     multiple_layers_combination_activation_rate, combination_map = \
            #         density.multiple_layers_combination_activation_rate(
            #             json_data[COV_FIRST], json_data[COV_SECOND])
            # except MemoryError:
            #     print('Multiple layers combination coverage rate ' + str(
            #         layer_name_list[json_data[COV_FIRST]])
            #           + str(layer_name_list[json_data[COV_SECOND]])
            #           + ': Out Of Memory Error')
            # else:
            #     print('Multiple layers combination coverage rate ' + str(
            #         layer_name_list[json_data[COV_FIRST]]) + ' and '
            #           + str(
            #         layer_name_list[json_data[COV_SECOND]]) + ': ' + str(
            #         multiple_layers_combination_activation_rate))
            # # Multiple layers combination coverage rate

            # Display heat map image in browser
            webbrowser.open(html_file_pass)

    finally:
        k.clear_session()
