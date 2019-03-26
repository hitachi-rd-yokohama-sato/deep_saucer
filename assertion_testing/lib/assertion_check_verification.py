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
import os

import random
import json


import numpy as np
from z3 import *

from pathlib import Path
from datetime import datetime

_root_dir = Path(__file__).absolute().parent.parent
_assertion_lib = Path(_root_dir).joinpath('lib')

sys.path.append(str(_root_dir))
sys.path.append(str(_assertion_lib))


from utils.structutil import NetworkStruct
from utils.z3util import VerifyZ3, get_z3obj_type

np.set_printoptions(linewidth=np.inf, precision=50)

_current = Path(os.getcwd()).absolute()

NAME_LIST = 'NameList'
PROP = 'Prop'
NUM_TEST = 'NumTest'

_required_conf_keys = [NAME_LIST, PROP]
_conf_keys = [NAME_LIST, PROP, NUM_TEST]

_conf_map = {
    NAME_LIST: lambda a: a[NAME_LIST],
    PROP: lambda a: a[PROP],
    NUM_TEST: lambda a: a[NUM_TEST] if NUM_TEST in a else None
}


def _hasattr_iter(o):
    return hasattr(o, '__iter__')


def _check_conf_json_keys(j_data):
    for conf_key in _required_conf_keys:
        if conf_key not in j_data:
            print('Config JSON not defined "{}"'.format(conf_key),
                  file=sys.stderr)
            return False
    return True


def _set_log_path():
    _log_dir = _current.joinpath('logfile')
    if not _log_dir.exists():
        _log_dir.mkdir(parents=True)

    return _log_dir.joinpath('log_{0}.txt'.format(
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))


def _get_conf_vals(j_data):
    for conf_key in _conf_keys:
        yield _conf_map[conf_key](j_data)


def _choice_dataset(dataset, input_var_placeholders, sample_num, num_test):
    random.seed(1)
    sample_index = random.sample(range(num_test), k=sample_num)

    test_dataset = []
    for i in range(len(dataset)):
        test_dataset.append([])
        for j in sample_index:
            if _hasattr_iter(dataset[i]):
                test_dataset[i].append(dataset[i][j].tolist())
            else:
                test_dataset[i] = dataset[i]

        if _hasattr_iter(test_dataset[i]):
            test_dataset[i] = np.array(
                test_dataset[i],
                dtype=input_var_placeholders[i].dtype.as_numpy_dtype)

    return test_dataset


def _check_shape(dataset, input_var_placeholders):
    for val, p, in zip(dataset, input_var_placeholders):
        if _hasattr_iter(val):
            if val.shape[1] != p.shape[1].value:
                print('{0} shape is {1}, data shape is {2}'.format(
                    p.name, p.shape, val.shape), file=sys.stderr)
                return False

    return True


def check_solver(vz3, in_data, out_data, smt_file_format):
    """
    Define and solve input and output values
    :param vz3:
    :param in_data:
    :param out_data:
    :param smt_file_format:
    :return:
    """
    for i, (i_data, o_data) in enumerate(zip(in_data, out_data)):
        # save base solver
        vz3.z3_obj.push()

        # Define variables
        _add_variable_value(i_data, o_data, vz3)

        yield vz3.z3_obj.check()

        with open(smt_file_format.format(i), 'w') as ws:
            print(vz3.z3_obj.to_smt2(), file=ws)

        # load base solver
        vz3.z3_obj.pop()


def _add_variable_value(in_data, out_data, vz3):
    if not isinstance(in_data, list):
        in_data = [in_data]
    if not isinstance(out_data, list):
        out_data = [out_data]
    # define input variable and value
    for in_d, ipi in zip(in_data, vz3.network_struct.input_placeholders_info):
        _def_var_val(in_d, ipi.dtype, ipi.var_names, vz3.z3_obj)

    # define output variable and value
    for out_d, opi in zip(out_data, vz3.network_struct.output_nodes_info):
        _def_var_val(out_d, opi.dtype, opi.var_names, vz3.z3_obj)


def _def_var_val(values, d_type, name_list, z3_obj):
    define_format = '{0}={1}("{0}")'
    eval_format = '{0}=={1}'

    if not hasattr(values, '__iter__'):
        value_list = [values]
    else:
        value_list = values

    for name, value in zip(name_list, value_list):
        tmp_expr_right = get_z3obj_type(d_type, name)
        tmp_expr = define_format.format(name, str(tmp_expr_right.sort()))
        exec(tmp_expr)

        # define value
        z3_obj.add(eval(eval_format.format(name, value)))


def _is_empty_file(path):
    """
    check empty
    :param path:
    :return:
    """
    with open(path, 'r') as rs:
        line = rs.readline()

    return line == ''


def main_p(model, name_list_path):
    log_file = _set_log_path()

    # Load DNN Structure
    network_struct = NetworkStruct()
    network_struct.load_struct(name_list_path)
    network_struct.set_info_by_session(model)
    network_struct.print_vars()

    with open(log_file, 'w') as ws:
        network_struct.print_vars(ws)

    print('Log File: {}'.format(str(log_file)))


def main(model, dataset=None, config_path=None):
    """

    :param model: Tensorflow graph session
    :param dataset: List of values given each input placeholder
    :param config_path:
    :return:
    """
    log_file = _set_log_path()

    num_test = max([data.shape[0] for data in dataset if _hasattr_iter(data)])

    # Load Config
    conf_dir = Path(config_path).parent.absolute()
    with open(config_path) as fs:
        j_data = json.load(fs)

    if not _check_conf_json_keys(j_data):
        return

    name_list_path, prop_path, sample_num = _get_conf_vals(j_data)

    if not os.path.isabs(name_list_path):
        name_list_path = str(conf_dir.joinpath(name_list_path).resolve())

    if not os.path.exists(name_list_path):
        print('[Error]: "{}" is not found'.format(name_list_path),
              file=sys.stderr)
        return

    if not os.path.isabs(prop_path):
        prop_path = str(conf_dir.joinpath(prop_path).resolve())

    if not os.path.exists(prop_path):
        print('[Error]: "{}" is not found'.format(prop_path),
              file=sys.stderr)
        return

    elif _is_empty_file(prop_path):
        print('[Error]: "{}" is empty'.format(prop_path), file=sys.stderr)
        return

    if sample_num is None or sample_num > num_test:
        sample_num = num_test

    # Load DNN Structure
    network_struct = NetworkStruct()
    ret = network_struct.load_struct(name_list_path)
    if not ret:
        return

    network_struct.set_info_by_session(model)

    # ----- Testing -----
    # Accumulator for Kappa Accuracy for imbalanced data
    input_var_placeholders = []
    for name in network_struct.in_raw_names:
        input_var_placeholders.append(model.graph.get_tensor_by_name(name))

    if sample_num != num_test:
        # choice dataset random
        dataset = _choice_dataset(
            dataset, input_var_placeholders, sample_num, num_test)

    # check shape
    if not _check_shape(dataset, input_var_placeholders):
        return

    model_output = model.graph.get_tensor_by_name(network_struct.out_raw_name)

    # Set feed_dict
    feed_dict = {}
    for i, (input_placeholder, value) in enumerate(
            zip(input_var_placeholders, dataset)):
        feed_dict[input_placeholder] = value

    # Run the training op
    prediction_output = model.run(model_output, feed_dict)

    vz3 = VerifyZ3(network_struct)
    vz3.parse_property(prop_path)

    # format for check solver
    input_dataset = []
    for i in range(len(prediction_output)):
        input_dataset.append([])
        for j in range(len(dataset)):
            if _hasattr_iter(dataset[j]):
                input_dataset[i].append(dataset[j][i])
            else:
                input_dataset[i].append(dataset[j])

    smt_dir = _current.joinpath('smt')
    if not smt_dir.exists():
        smt_dir.mkdir(parents=True)

    smt_file_format = str(smt_dir.joinpath("check_{}.smt"))

    msg_map = {
        z3.sat.r: 'violated',
        z3.unsat.r: 'preserved'
    }

    with open(log_file, 'w') as ws:
        for i, r in enumerate(list(check_solver(vz3, input_dataset,
                                                prediction_output,
                                                smt_file_format))):

            out_str = '{0} : {1}'.format(i, msg_map[r.r])
            print(out_str)
            print(out_str, file=ws)

    print('Log File: {}'.format(str(log_file)))

    del (dataset, prediction_output, feed_dict)
