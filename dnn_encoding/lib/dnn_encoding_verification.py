# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import sys
import os
import json
from z3 import *
from pathlib import Path


_root_dir = Path(__file__).absolute().parent.parent
_lib_dir = Path(_root_dir).joinpath('lib')

sys.path.append(str(_root_dir))
sys.path.append(str(_lib_dir))

from utils.structutil import NetworkStruct
from utils.smtUtil import (
    make_smt_file, get_satisfiable, read_smt_file, make_smt_solver)
from utils.z3util import VerifyZ3


NAME_LIST = 'NameList'
PROP = 'Prop'
CONDITION = 'Condition'

_required_conf_keys = [NAME_LIST, PROP]
_conf_keys = [NAME_LIST, PROP, CONDITION]

_conf_map = {
    NAME_LIST: lambda a: a[NAME_LIST],
    PROP: lambda a: a[PROP],
    CONDITION: lambda a: a[CONDITION] if CONDITION in a else None
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


def _get_conf_vals(j_data):
    for conf_key in _conf_keys:
        yield _conf_map[conf_key](j_data)


def check_solver(model_smt_path, prop_smt_path, cond_smt_path, debug=False):
    if debug:
        print("start method checkSolver")

    s = z3.Solver()

    read_smt_file(s, model_smt_path, debug=debug)
    read_smt_file(s, prop_smt_path, debug=debug)

    if cond_smt_path is not None:
        read_smt_file(s, cond_smt_path, debug=debug)

    if debug:
        print(s.to_smt2())

    c = s.check()

    if debug:
        print("end method checkSolver")

    return c, s


def _is_empty_file(path):
    """
    check empty
    :param path:
    :return:
    """
    with open(path, 'r') as rs:
        line = rs.readline()

    return line == ''


def _check_network_shape(ns):
    nodes = ns.input_placeholders_info +\
            ns.hidden_nodes_info +\
            ns.output_nodes_info

    in_val = None
    out_val = None
    for n in nodes:
        if len(n.shape) != 2:
            print('shape {} is not supported'.format(n.shape),
                  file=sys.stderr)
            return False
        i, o = n.shape
        if in_val is None and out_val is None:
            in_val = i
            out_val = o

        elif out_val != i:
            print('No connection next input and output.\n'
                  '(output: {0}, next input: {1})'.format(out_val, i),
                  file=sys.stderr)
            return False
        else:
            in_val = i
            out_val = o

    return True


def main(model, dataset=None, config_path=None):
    # Load Config
    conf_dir = Path(config_path).parent.absolute()
    with open(config_path) as fs:
        j_data = json.load(fs)

    if not _check_conf_json_keys(j_data):
        return

    name_list_path, prop_path, condition_path = _get_conf_vals(j_data)

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

    if condition_path is not None:
        if not os.path.isabs(condition_path):
            condition_path = str(conf_dir.joinpath(condition_path).resolve())

        if not os.path.exists(condition_path):
            print('[Error]: "{}" is not found'.format(condition_path),
                  file=sys.stderr)
            return
        elif _is_empty_file(condition_path):
            condition_path = None

    current = Path(os.getcwd()).absolute()

    # Load DNN Structure
    network_struct = NetworkStruct()
    ret = network_struct.load_struct(name_list_path)
    if not ret:
        return

    network_struct.set_info_by_session(model)

    if not _check_network_shape(network_struct):
        return

    vz3_model = VerifyZ3(network_struct)
    make_smt_solver(vz3_model.z3_obj, vz3_model.network_struct)

    model_smt_path = str(Path(current, 'smt', 'model.smt'))
    # output model smt file
    make_smt_file(vz3_model.z3_obj, model_smt_path)
    print(model_smt_path)

    vz3_prop = VerifyZ3(network_struct)
    # read SMT-LIB2.0 format file (property)
    vz3_prop.parse_property(prop_path)

    prop_smt_path = str(Path(current, 'smt', 'property.smt'))
    # output property smt file
    make_smt_file(vz3_prop.z3_obj, prop_smt_path)
    print(prop_smt_path)

    cond_smt_path = None
    if condition_path is not None:
        vz3_cond = VerifyZ3(network_struct)
        # read SMT-LIB2.0 format file (condition)
        vz3_cond.parse_condition(condition_path)

        cond_smt_path = str(Path(current, 'smt', 'condition.smt'))
        # output condition smt file
        make_smt_file(vz3_cond.z3_obj, cond_smt_path)
        print(cond_smt_path)

    # check sat
    check_smt_path = str(Path(current, 'smt', 'check.smt'))
    print(check_smt_path)
    c, s3 = check_solver(model_smt_path, prop_smt_path, cond_smt_path)
    with open(check_smt_path, 'w') as ws:
        print(s3.to_smt2(), file=ws)

    # output solver answer
    satisfiable_path = str(Path(current, 'sat', 'satisfiable.txt'))
    get_satisfiable(s3, network_struct, satisfiable_path)
