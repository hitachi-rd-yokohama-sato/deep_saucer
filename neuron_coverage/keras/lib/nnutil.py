# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import functools
import operator
import types
from abc import ABCMeta, abstractmethod
from collections.abc import Sequence

import keras.backend as k
import numpy as np
from keras.layers import Activation

from lib.activations import Activation as Act
import pandas as pd
import shutil
from datetime import datetime as dt
import os


class Density:
    def __init__(self, graph):
        self.graph = graph
        num_layers = len(graph.networks[0])
        self.result = [0] * num_layers

    def update(self, activation_value):
        for network in self.graph:
            weight = network.weight
            for layer in network.layers:
                layer.activated_output = layer.activated_output.astype("float32")

                if activation_value == 0:
                    pass
                elif activation_value == 1:
                    layer.activated_output = layer.activated_output / weight

        for network in self.graph:
            for index, layer in enumerate(network.layers):
                self.result[index] += layer.activated_output

    def same_layers_combination_activation_rate(self, layer_no):
        graph_network_layer = self.graph.networks[0].layers[layer_no]
        # same layer combination
        size = graph_network_layer.activated_output[0].shape[0]
        all_sum = size * (size-1) / 2
        combination_map = np.zeros((size, size), dtype=np.float32)
        for idx, network in enumerate(self.graph):

            # replace activated 1
            activated_output = network.layers[layer_no].activated_output[0]
            activated_output[activated_output != 0] = 1
            # create combination map
            combination_map += np.outer(
                activated_output, activated_output).astype(np.float32)

        # combination coverage
        combination_activate_sum = 0
        for i, p in enumerate(combination_map):
            combination_activate_sum += np.count_nonzero(p[0:i])

        combination_coverage_rate = combination_activate_sum/all_sum
        return combination_coverage_rate, combination_map

    def multiple_layers_combination_activation_rate(
            self, first_layer_no, second_layer_no):

        if first_layer_no == second_layer_no:
            self.same_layers_combination_activation_rate(first_layer_no)

        gn_first_layer = self.graph.networks[0].layers[first_layer_no]
        gn_second_layer = self.graph.networks[0].layers[second_layer_no]

        first_layer_size = gn_first_layer.activated_output[0].shape[0]
        second_layer_size = gn_second_layer.activated_output[0].shape[0]
        all_sum = first_layer_size * second_layer_size

        combination_map = np.zeros((first_layer_size, second_layer_size),
                                   dtype=np.float32)

        for idx, network in enumerate(self.graph):
            # replace activated 1
            first_layer = network.layers[first_layer_no]
            second_layer = network.layers[second_layer_no]

            first_layer_activated_output = first_layer.activated_output[0]
            second_layer_activated_output = second_layer.activated_output[0]

            first_layer_activated_output[first_layer_activated_output != 0] = 1
            second_layer_activated_output[second_layer_activated_output != 0] = 1

            # create combination map
            combination_map += np.outer(
                first_layer_activated_output,
                second_layer_activated_output).astype(np.float32)
            combination_map[combination_map != 0] = 1

        combination_coverage_rate = np.count_nonzero(combination_map)/all_sum
        return combination_coverage_rate, combination_map


class Coverage:
    @staticmethod
    def activation_rate(network):
        """
        :type network: Network
        :rtype: np.ndarray
        """
        activation_counters = network.activation_counters
        _counter = 0
        _size = 0
        for counter in activation_counters:
            _counter += np.count_nonzero(counter)
            _size += counter.size
        return _counter / _size

    @staticmethod
    def inactivation_rate(network):
        """
        :type network: Network
        :rtype: np.ndarray
        """
        inactivation_counters = network.inactivation_counters
        _counter = 0
        _size = 0
        for counter in inactivation_counters:
            _counter += np.count_nonzero(counter)
            _size += counter.size
        return np.array([_counter / _size])


class IActivationResult(metaclass=ABCMeta):
    @abstractmethod
    def activation_counters(self):
        """ return at least one non_activated counters per layer.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def inactivation_counters(self):
        """ return at least one non_activated counters per layer.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def get_input_count(self):
        """ return the number of input shapes.
        :rtype: int
        """
        pass

    @abstractmethod
    def get_unit_count(self):
        """ return the number of all units.
        :rtype: int
        """
        pass


class Graph(Sequence):
    def __init__(self, model, layer_filters, inputs,
                 determination_on_activation_kw,
                 activation_value_add_neuron_kw):
        """
        :type model: keras.model
        :type layer_filters: list[keras.core.layers]
        :type inputs: np.ndarray or list[np.ndarray]
        """
        self.layer_filters = layer_filters
        # self.input = inp
        self.networks = [Network(model, layer_filters, inp,
                                 determination_on_activation_kw,
                                 activation_value_add_neuron_kw)
                         for inp in inputs]

    def __len__(self):
        return len(self.networks)

    def __getitem__(self, index):
        return self.networks[index]


class Network(IActivationResult, Sequence):
    def __init__(self, model, layer_filters, model_input,
                 determination_on_activation_kw,
                 activation_value_add_neuron_kw):
        """
        :type model: keras.model
        :type layer_filters: list[keras.core.layers]
        :type model_input: np.ndarray or list[np.ndarray]
        """
        if isinstance(model_input, np.ndarray):
            pass
        elif isinstance(model_input, types.GeneratorType):
            pass
        elif isinstance(model_input, list):
            self.input_shape = model_input[0].shape[0]
        else:
            raise TypeError()
        self.all_units = []
        self.layer_filters = layer_filters
        self.input = model_input

        if len(self.layer_filters) > 0:
            self.layers = [self.set_layer(layer, determination_on_activation_kw,
                                          activation_value_add_neuron_kw)
                           for layer in model.layers
                           if is_any_layer(layer, self.layer_filters)]
        else:
            self.layers = [self.set_layer(layer, determination_on_activation_kw,
                                          activation_value_add_neuron_kw)
                           for layer in model.layers
                           if hasattr(layer, 'activation')]
        self.update()
        self.weight = np.sum([layer.weight for layer in self.layers])

    @staticmethod
    def set_layer(layer, determination_on_activation_kw,
                  activation_value_add_neuron_kw):

        activation_func_name = layer.activation.__name__
        _layer = Layer(layer.name, activation_func_name, layer.output_shape)
        _layer.activation_func = Act(activation_func_name,
                                     determination_on_activation_kw,
                                     activation_value_add_neuron_kw).activation
        return _layer

    def update(self):
        """
        Update input value of model and layers
        """
        self.all_units = []
        for layer, output in zip(self.layers, self.input):
            layer.update(output)
            self.all_units.append(layer.units_num)

    @property
    def activation_counters(self):
        return self.get_activation_counters_per_layers

    @property
    def inactivation_counters(self):
        return self.get_inactivation_counters_per_layers

    @property
    def get_activation_counters_per_layers(self):
        """ return at least one activated counters per layer.
        :rtype: tuple(np.ndarray)
        """
        return ((np.sum(layer.activated_output, axis=0)
                 for layer in self.layers))

    @property
    def get_inactivation_counters_per_layers(self):
        """ return at least one non_activated counters per layer.
        :rtype: tuple(np.ndarray)
        """
        return ((self.input_shape - np.sum(layer.activated_output, axis=0)
                 for layer in self.layers))

    def get_unit_count(self):
        pass

    def get_input_count(self):
        pass

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)


class ALayer(object):
    pass


class HiddenLayer(ALayer):
    pass


class OutputLayer(ALayer):
    pass


class Layer(IActivationResult):
    def __init__(self, layer_name, activation_function_name, output_shapes):
        """
        :type layer_name: str
        :type activation_function_name: str
        :type output_shapes: tuple
        """
        self.name = layer_name
        self.activation = activation_function_name
        self.output_shapes = output_shapes
        self.input_shapes = 0
        self.units_num = functools.reduce(operator.mul, output_shapes[1:], 1)
        self.activation_func = None
        self.activated_output = None
        self.weight = 0

    def update(self, model_input):
        """ Execute this class method each time input changes.
        It will be set activated_outputs.
        When the layer name is "conv",
        activated_outputs is reshaped to flatten outputs.

        ex: input_shape (1, 32, 32, 32) -> output_shape (1, 32768)

        :type model_input: np.ndarray
        """
        self.input_shapes = model_input.shape[0]
        if self.activation_func:
            activated_output = self.activation_func(model_input)
            if 'conv' in self.name.lower():
                activated_output = activated_output.reshape(
                    [self.input_shapes, self.units_num])
                # channel first
                self.activated_output = activated_output
            else:
                # channel first
                self.activated_output = activated_output

        else:
            raise TypeError

        self.weight = np.sum(activated_output)

    @property
    def activation_counters(self):
        return np.sum(self.activated_output, axis=0)

    @property
    def inactivation_counters(self):
        return self.input_shapes - np.sum(self.activated_output, axis=0)

    def get_unit_count(self):
        return self.input_shapes

    def get_input_count(self):
        return self.input_shapes

    def __str__(self):
        return self.name


class Unit:
    def __init__(self):
        self.weight = 0
        self.activation = False

    def update(self, weight, activation):
        self.weight += weight
        self.activation = activation


class Dataset:
    def __init__(self, model, layer_filters):
        # check model
        validate_not_include_in_activation_layer(model)
        self.funcs, self.multi_inputs = get_intermediate_funcs(model,
                                                               layer_filters)
        self._inputs = None

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        if self.multi_inputs:
            list_inputs = []
            list_inputs.extend(inputs)
            list_inputs.append(0.)
        else:
            list_inputs = [inputs, 0.]
        self._inputs = list_inputs

    def activated_outputs(self):
        for func in self.funcs:
            yield func(self._inputs)[0]

    def layer_activated_outputs(self):
        list_inputs = []
        if not self.multi_inputs:
            for model_input in self._inputs[0]:
                list_inputs.append([[model_input], 0.])

        for list_input in list_inputs:
            yield [func(list_input)[0] for func in self.funcs]


def is_any_layer(layer, layer_filters):
    """
    :type layer: object
    :type layer_filters: list[keras.layers, ]
    :rtype: bool

    >>> from keras.layers import Activation, Conv2D, Dense
    >>> is_any_layer(Dense(64), [Activation, Conv2D, Dense])
    True
    >>> is_any_layer(Dense(64), [Activation, Conv2D])
    False
    """

    if {type(layer)} & set(layer_filters):
        return True
    else:
        return False


def is_target_layer(layer, layer_filters):
    if not hasattr(layer, 'activation'):
        return False

    elif len(layer_filters) > 0 and not is_any_layer(layer, layer_filters):
        return False
    else:
        return True


def validate_not_include_in_activation_layer(model):
    """ Check whether model includes in activation layer or not.

    :type model: keras.model
    """
    for layer in model.layers:
        if is_any_layer(layer, [Activation]):
            raise AttributeError("This model includes in activation layer!!"
                                 "Try to use a model that includes "
                                 "the activation function as an argument")


def get_outputs_per_layer(model, layer_filters, model_inputs):
    """ Calculate the output of all layers using
     the weights and biases of existing learned models.

    :type model: keras.model
    :type layer_filters: list[keras.layers,]
    :type model_inputs: np.ndarray
    :rtype: list[np.ndarray]
    """
    # check model
    validate_not_include_in_activation_layer(model)

    funcs, model_multi_inputs_cond = get_intermediate_funcs(model,
                                                            layer_filters)

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]

    layer_outputs = []
    for func in funcs:
        print('Calculate output : {0}'.format(str(func.outputs)))
        layer_outputs.append(func(list_inputs)[0])

    return layer_outputs


def get_outputs_per_input(model, layer_filters, model_inputs):
    """ Calculate the output of all layers using
    the weights and biases of existing learned models.

    :type model: keras.model
    :type layer_filters: list[keras.layers,]
    :type model_inputs: np.ndarray
    :rtype: list[np.ndarray]
    """

    validate_not_include_in_activation_layer(model)

    funcs, model_multi_inputs_cond = get_intermediate_funcs(model,
                                                            layer_filters)

    list_inputs = []
    if not model_multi_inputs_cond:
        for model_input in model_inputs:
            list_inputs.append([[model_input], 0.])

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = []
    for list_input in list_inputs:
        layer_outputs.append([func(list_input)[0] for func in funcs])

    return layer_outputs


def get_intermediate_funcs(model, layer_filters):
    """ Generate a function that outputs calculation result of middle layer.

    :type model: keras.model
    :type layer_filters: list[keras.layers,]
    :rtype: list[keras.backend.tensorflow_backend.Function], bool
    """
    inp = model.input
    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    if len(layer_filters) > 0:
        outputs = [layer.output for layer in model.layers
                   if is_any_layer(layer, layer_filters)]
    else:
        outputs = [layer.output for layer in model.layers
                   if hasattr(layer, 'activation')]

    # evaluation functions
    funcs = [k.function(inp + [k.learning_phase()], [out])
             for out in outputs]

    return funcs, model_multi_inputs_cond


def density_to_heatmap_html(density, activation_value_add_neuron_variation):
    dfs = []

    file_pass = os.path.dirname(__file__)

    timestamp = dt.now().strftime('%Y%m%d%H%M%S%f')
    file_name = "coverage_" + timestamp + ".html"

    for layer_no, layer in enumerate(density.result, 1):
        layer_name = density.graph.networks[0].layers[layer_no - 1].name
        layer_name_str = "\'" + layer_name + "\'"
        d = pd.DataFrame(
            {"layer": [layer_no for _ in range(layer.size)],
             "layer_name": [layer_name_str for _ in range(layer.size)],
             "unit": [u for u in range(layer.size)],
             "value": [unit for unit in layer[0]]
             })

        if activation_value_add_neuron_variation is 1:
            d["value"] = [1 if v > 0 else v for v in d["value"]]
        else:
            d["value"] = d["value"].round(10)

        dfs.append(d)

    master_df = pd.concat(dfs, ignore_index=True)
    pd.DataFrame(master_df).to_csv(file_name, index=False)

    # # debug
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # test_list = []
    # for test in density.result:
    #     test_list.append(np.asarray(test).reshape(-1))
    #
    # df3 = pd.DataFrame(test_list)
    # plt.figure()
    # sns.heatmap(df3.T, cmap='coolwarm')
    # plt.show()
    # # debug

    f = open(file_name)
    reline = f.readlines()
    f.close()
    new_val = []
    first_str = "\t\t\tvar in_data = [\n"
    last_str = "\t\t\t];\n"
    for idx, val in enumerate(reline, 0):
        val = val.replace('\n', '')
        if len(reline) == idx + 1:
            new_val_st = "\t\t\t[" + val + "]\n"
        else:
            new_val_st = "\t\t\t[" + val + "],\n"
        new_val.append(new_val_st)
    new_val[0] = first_str
    new_val.append(last_str)

    t = open(file_pass + '/heatMap/temp/templeate.html')
    html_str = t.readlines()
    t.close()

    w = open(file_name, "w")
    for idx1, val1 in enumerate(html_str, 0):

        if val1 == "\t\t\t// @Data setting position\n":
            for idx2, val2 in enumerate(new_val, 0):
                w.write(val2)
        else:
            w.write(val1)
    w.close()

    # save heatMap directory
    shutil.move(file_name, file_pass + '/heatMap')

    return file_pass + "/heatMap/" + file_name
