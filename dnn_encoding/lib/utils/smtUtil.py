#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************

from lib.utils.funcDef import *
from lib.utils.z3util import get_z3obj_type


msg_map = {
        z3.sat.r: 'violated',
        z3.unsat.r: 'preserved'
    }


def make_smt_solver(solver, network_struct):
    # create input information
    input_list = []
    for input_placeholder in network_struct.input_placeholders_info:
        for var_name in input_placeholder.var_names:
            tmp = get_z3obj_type(input_placeholder.dtype, var_name)
            input_list.append(tmp)

    # create layer information
    node_list = []
    node_list.extend(network_struct.hidden_nodes_info)
    node_list.extend(network_struct.output_nodes_info)

    for node in node_list:

        tmp_in_list = []
        tmp_exp = []
        tmp_sum_exp = ""

        for j in range(len(node.var_names)):

            tmp_layer = get_z3obj_type(node.dtype, node.var_names[j])
            tmp_in_list.append(tmp_layer)

            tmp_add = node.b.tolist()[j]

            for k in range(len(input_list)):
                tmp_add = tmp_add + node.W.tolist()[k][j] * input_list[k]

            if node.func == "Softmax":
                tmp = calcExp(tmp_add)
                tmp_exp.append(tmp)
                tmp_sum_exp = tmp_sum_exp + tmp
            else:
                tmp_add = eval(node.func)(tmp_add)
                solver.add(tmp_layer == tmp_add)

        if node.func == "Softmax":
            for j in range(len(tmp_in_list)):
                solver.add(tmp_in_list[j] == tmp_exp[j] / tmp_sum_exp)

        input_list = tmp_in_list


def make_smt_file(s, path, debug=False):

    if debug:
        print("start method make_smt_file")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        f.write(s.to_smt2())

    if debug:
        print("end method make_smt_file")


def read_smt_file(s, path, debug=False):

    if debug:
        print("start method read_smt_file")

    if os.path.exists(path):
        s.from_file(path)
    else:
        print("file not found:" + path)

    if debug:
        print("end method read_smt_file")


def get_satisfiable(s, network_struct, path, debug=False):

    if debug:
        print("start method get_satisfiable")

    # check sat
    c = s.check()

    str_ = ''

    if c == z3.sat:

        m = s.model()
        # join name list input layer and output layer
        inout_list = []
        for name in network_struct.in_var_names:
            inout_list.extend(name)

        for name in network_struct.out_var_names:
            inout_list.extend(name)

        ans_dict = {
            ans.name(): ans for ans in m if ans.name() in inout_list
        }
        # get name
        # ex) 'x:0.1'
        lst = list(
            map(lambda n: '{0}:{1}'.format(n, m[ans_dict[n]]), inout_list)
        )

        str_ = ','.join(lst)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(str_)
        print(path)

    print('--------------------')
    print('Property is {}'.format(msg_map[c.r]))
    if str_ != '':
        print('Counterexample is')
        print(str_)
    print('--------------------')

    if debug:
        print("end method get_satisfiable")
