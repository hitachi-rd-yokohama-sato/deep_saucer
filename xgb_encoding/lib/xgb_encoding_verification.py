import sys
import pickle
import pandas as pd
import json

from z3 import *
import copy
import time
import datetime
import math

from xgboost.core import *
from xgboost.core import _check_call
from xgboost.core import _LIB

import random
from pathlib import Path
_root_dir = Path(__file__).absolute().parent.parent
_lib_dir = _root_dir.joinpath('lib')

sys.path.append(str(_root_dir))
sys.path.append(str(_lib_dir))

import z3util

import traceback

_LOG_MSG_INPUT_VALUE = '  input to {0} is {1}'
_LOG_MSG_OUTPUT_VALUE = '  output from {0} is {1}\n'
_LOG_MSG_WARN_VARIAVLES = 'Warning: Explanatory variable "{}" ' \
                          'not used in model.'
_LOG_MSG_WARN_UPPER_LIMIT = 'Warning: Explanatory variable "{}" ' \
                            'was not set Upper limit.' \
                            'Search violation may not stop.'
_LOG_MSG_WARN_LOWER_LIMIT = 'Warning: Explanatory variable "{}" ' \
                            'was not set Lower limit.' \
                            'Search violation may not stop.'

_SEARCH_RANGE_RATIO = 'search_range_ratio'
_BUFFER_RANGE_RATIO = 'buffer_range_ratio'
_PROP_PATH = 'prop_path'
_DATA_LIST_PATH = 'data_list_path'
_COND_PATH = 'cond_path'
_SYSTEM_TIMEOUT = 'system_timeout'
_SEARCH_TIMEOUT = 'search_timeout'
_EXTEND_ARG = 'extend_arg'
_SAMETIME = 'sametime'
_SEPARATE = 'separate'
_SPLITTING = 'splitting'
_VOL_RATIO = 'vol_ratio'
_PERMUTATION_LIST_MAX = 'permutation_list_max'


_INPUT = 'input'
_OUTPUT = 'output'

_NAME = 'name'
_CONT_VALUE_FLAG = 'cont_value_flag'
_TYPE = 'type'
_UPPER = 'upper'
_LOWER = 'lower'

_Z3 = 'z3'
_XGBOOST = 'xgboost'
_DIRECTION = 'direction'

num_call_solver = [0, 0.0]


def check_solver(solver):
    num_call_solver[0] += 1
    time_before_check = time.time()
    result = solver.check()
    num_call_solver[1] += (time.time() - time_before_check)

    return result


def get_dump(booster, fmap='', with_stats=False, dump_format="json"):
    """
    Returns the dump the model as a list of strings.
    """

    length = c_bst_ulong()
    sarr = ctypes.POINTER(ctypes.c_char_p)()
    if booster.feature_names is not None and fmap == '':
        flen = len(booster.feature_names)

        fname = from_pystr_to_cstr(booster.feature_names)

        if booster.feature_types is None:
            # use quantitative as default
            # {'q': quantitative, 'i': indicator}
            ftype = from_pystr_to_cstr(['q'] * flen)
        else:
            ftype = from_pystr_to_cstr(booster.feature_types)
        _check_call(_LIB.XGBoosterDumpModelExWithFeatures(
            booster.handle,
            ctypes.c_int(flen),
            fname,
            ftype,
            ctypes.c_int(with_stats),
            c_str(dump_format),
            ctypes.byref(length),
            ctypes.byref(sarr)))
    else:
        if fmap != '' and not os.path.exists(fmap):
            raise ValueError("No such file: {0}".format(fmap))
        _check_call(_LIB.XGBoosterDumpModelEx(booster.handle,
                                              c_str(fmap),
                                              ctypes.c_int(with_stats),
                                              c_str(dump_format),
                                              ctypes.byref(length),
                                              ctypes.byref(sarr)))
    res = from_cstr_to_pystr(sarr, length)
    return res


def dump_model(booster, fout, fmap='', with_stats=False, dump_format="json"):
    """
    Dump model into a text file.

    Parameters
    ----------
    booster :
    fout : string
        Output file name.
    fmap : string, optional
        Name of the file containing feature map names.
    with_stats : bool (optional)
        Controls whether the split statistics are output.
    dump_format : string, optional
            Format of model dump file. Can be 'text' or 'json'.
    """
    if isinstance(fout, STRING_TYPES):
        fout = open(fout, 'w')
        need_close = True
    else:
        need_close = False
    ret = get_dump(booster, fmap, with_stats, dump_format=dump_format)
    if dump_format == 'json':
        fout.write('[\n')
        for i in range(len(ret)):
            fout.write(ret[i])
            if i < len(ret) - 1:
                fout.write(",\n")
        fout.write('\n]')
    else:
        for i in range(len(ret)):
            fout.write('booster[{}]:\n'.format(i))
            fout.write(ret[i])
    if need_close:
        fout.close()


def dec_exp_vars_rev(input_type):

    input_array = []
    for i in range(len(input_type)):

        if 'int' in str(input_type[i]):
            input_array.append(Int(_INPUT + str(i)))
        elif 'float' in str(input_type[i]):
            input_array.append(Real(_INPUT + str(i)))
        else:
            assert False

    return input_array


def input_name_replacer(tree, input_name):

    if 'split' in tree:
        name_var = tree['split']
        success = False
        for index, n in enumerate(input_name):
            if name_var == n:
                tree['split'] = 'input' + str(index)
                success = True
                break

        if success is False:
            print('Error ' + name_var + ' could not be repleced')
            assert False

    if 'children' in tree:
        for child in tree['children']:
            input_name_replacer(child, input_name)


# def fname_replacer(fname, input_name):
#     var_index = int(fname[5:])
#     return input_name[var_index]


def add_constraints(input_array, input_type, out_var, tree, antecedent,
                    solver, input_used_or_not):

    if 'split' in tree and 'split_condition' in tree and \
            'yes' in tree and 'no' in tree and 'children' in tree:
        var_str = tree['split']
        var_index = int(var_str[5:])

        if var_index < len(input_used_or_not):
            input_used_or_not[var_index] = True
        else:
            print('Error: the length of input_used_or_not is invalid')
            assert False

        var = input_array[var_index]
        val_original = tree['split_condition']
        if 'int' in str(input_type[var_index]):
            val = math.ceil(val_original)
        else:
            val = val_original
        children_trees = tree['children']
        for child_tree in children_trees:
            if child_tree['nodeid'] == tree['yes']:
                cond = var < val
                antecedent_copy = copy.deepcopy(antecedent)
                antecedent_copy.append(cond)
                add_constraints(input_array, input_type, out_var,
                                child_tree, antecedent_copy, solver,
                                input_used_or_not)

            if child_tree['nodeid'] == tree['no']:
                cond = var >= val
                antecedent_copy = copy.deepcopy(antecedent)
                antecedent_copy.append(cond)
                add_constraints(input_array, input_type, out_var,
                                child_tree, antecedent_copy, solver,
                                input_used_or_not)

    elif 'leaf' in tree:
        if tree['leaf'] == -0:
            leaf_val = 0
        else:
            leaf_val = tree['leaf']

        if len(antecedent) == 0:
            solver.add(out_var == leaf_val)
        else:
            antece_formula = And([cond for cond in antecedent])
            solver.add(Implies(antece_formula, out_var == leaf_val))

    else:
        print("Unexpected Error: A decision tree comprising "
              "the XGBoost model has an unexpected attribute")
        assert False  # TODO Error handling


def add_trees_consts_to_solver(djson, input_array, input_type, solver,
                               input_used_or_not):

    sub_c_array = []
    for tree_id, tree in enumerate(djson):

        out_var = Real('c{0}'.format(tree_id))

        sub_c_array.append(out_var)
        antecedent = []
        add_constraints(input_array, input_type, out_var, tree, antecedent, solver,
                        input_used_or_not)
    return sub_c_array


def add_trees_relation_regressor(sub_c_array, solver):
    c_sum = Real('c_sum')
    added = sub_c_array[0]
    for index in range(1, len(sub_c_array)):
        added = added + sub_c_array[index]

    rel_formula = (c_sum == added)
    solver.add(rel_formula)

    return c_sum


# def create_upperlower_from_input(input):
#
#     upper_limit_str = []
#     lower_limit_str = []
#     for col_index in range(len(input.columns)):
#         col_data = input.iloc[:, [col_index]]
#         upper_limit_str.append(str(col_data.max()[0]))
#         lower_limit_str.append(str(col_data.min()[0]))
#
#     return lower_limit_str, upper_limit_str


# def parse_upperlower_file(upper_lower_file):
#     with open(upper_lower_file, 'r') as rs:
#         for line in rs.readline():
#             choped_line = line.strip()


def add_upperlower_constraints(input_array, solver, upper_limit, lower_limit):
    print_solver = Solver()
    for f_index in range(len(input_array)):
        # 上限の追加
        upper = upper_limit[f_index]
        if upper is not None:
            upper_constraint = input_array[f_index] <= upper
            solver.add(upper_constraint)
            print_solver.add(upper_constraint)
        else:
            pass

        lower = lower_limit[f_index]

        if lower is not None:
            lower_constraint = input_array[f_index] >= lower
            solver.add(lower_constraint)
            print_solver.add(lower_constraint)

    _log_writer('===== upper lower constraints =====', log_file)
    _log_writer(print_solver.sexpr(), log_file)

    del print_solver


def add_other_constraints(input_array, output_array, in_names, out_names,
                          solver, constraint_file):

    constraint = z3util.parse_constraint(constraint_file, in_names, out_names)
    solver.add(eval(constraint))

    print_solver = Solver()
    print_solver.add(eval(constraint))
    _log_writer('===== other constraints =====', log_file)
    _log_writer(print_solver.sexpr(), log_file)

    del print_solver


def add_property(input_array, output_array,
                 in_names, out_names, solver, constraint_file):

    prop_str = z3util.parse_constraint(constraint_file, in_names, out_names)
    neg_prop = Not(eval(prop_str))
    solver.add(neg_prop)

    print_solver = Solver()
    print_solver.add(neg_prop)
    _log_writer('===== property =====', log_file)
    _log_writer(print_solver.sexpr(), log_file)

    del print_solver


def cast_upperlower(lower_limit, upper_limit, input_type):
    upper_limit_list = []
    lower_limit_list = []
    for f_index in range(len(input_type)):

        if upper_limit[f_index] is not None:
            if 'int' in str(input_type[f_index]):
                try:
                    upper = int(upper_limit[f_index])
                except:
                    print("Unexpected value is given as upper_bound")
                    assert False
            elif 'float' in str(input_type[f_index]):
                try:
                    upper = float(upper_limit[f_index])
                except:
                    print("Unexpected value is given as upper_bound")
                    assert False
            else:
                print("Unexpected input_type is designated")
                assert False

            upper_limit_list.append(upper)

        else:
            upper_limit_list.append(None)

        if lower_limit[f_index] is not None:
            if 'int' in str(input_type[f_index]):
                try:
                    lower = int(lower_limit[f_index])
                except:
                    print("Unexpected value is given as lower_bound")
                    assert False
            elif 'float' in str(input_type[f_index]):
                try:
                    lower = float(lower_limit[f_index])
                except:
                    print("Unexpected value is given as lower_bound")
                    assert False
            else:
                print("Unexpected input_type is designated")
                assert False

            lower_limit_list.append(lower)
        else:
            lower_limit_list.append(None)

    return lower_limit_list, upper_limit_list


def calc_diff(lower_limit, upper_limit, input_type, search_range_ratio):
    if 'int' in str(input_type):
        if lower_limit is None:
            if upper_limit is None:
                diff = 1
            else:
                diff = math.ceil(int(abs(upper_limit) / search_range_ratio))
        else:
            if upper_limit is None:
                diff = math.ceil(int(abs(lower_limit) / search_range_ratio))
            else:
                diff = math.ceil(
                    (int(upper_limit) - int(lower_limit)) / search_range_ratio)

        if diff < 1:
            diff = 1

    elif 'float' in str(input_type):
        if lower_limit is None:
            if upper_limit is None:
                diff = 0.1
            else:
                diff = float(abs(upper_limit) / search_range_ratio)
        else:
            if upper_limit is None:
                diff = float(abs(lower_limit) / search_range_ratio)
            else:
                diff = float((float(upper_limit) - float(
                    lower_limit)) / search_range_ratio)

        if diff == 0:
            diff = 0.1

    else:
        print("Error: input_type is not defined")
        assert False

    return diff


def find_extents_rev(solver, model, lower_limit, upper_limit,
                     input_array, output_array, input_type,
                     model_array, cont_value_flag, search_range_ratio,
                     input_name, input_used_or_not,
                     start_time, system_time_out, search_time_out, extend_arg):

    find_start = time.time()

    base_volume = calc_volume(lower_limit, upper_limit)
    input_array_solution = [None] * len(input_array)
    m = solver.model()
    for index in range(len(input_array)):
        z3obj = m[input_array[index]]
        if z3obj is not None:
            if 'int' in str(input_type[index]):
                num = int(z3obj.as_long())
            elif 'float' in str(input_type[index]):
                num = float(z3obj.as_fraction())
            else:
                print("Error: input_type is not defined")
                assert False

            input_array_solution[index] = num

    upper_extent = [0] * len(input_array)
    lower_extent = [0] * len(input_array)
    no_solution_spaces_fixed = []

    no_solution_space_tmp = {}

    for index in range(len(input_array_solution)):
        if input_array_solution[index] is not None:
            upper_extent[index] = input_array_solution[index]
            lower_extent[index] = input_array_solution[index]
        else:
            upper_extent[index] = 0
            lower_extent[index] = 0

    solver.push()

    diff_list = []
    for index in range(len(input_array)):
        diff = calc_diff(lower_limit[index], upper_limit[index],
                         input_type[index], search_range_ratio)
        diff_list.append(diff)

    flag = True
    search_count = 0
    while flag is True:
        flag = False

        if extend_arg == _SAMETIME:
            solver.push()

            new_upper = copy.deepcopy(upper_extent)
            new_lower = copy.deepcopy(lower_extent)
            for index in range(len(input_array)):
                if (cont_value_flag[index]) and (
                        input_array_solution[index] is not None) and (
                        input_used_or_not[
                            index] is True):

                    if (((upper_limit[index] is not None) and (
                            upper_extent[index] == upper_limit[index])) or (
                            diff_list[index] == 0)) and \
                            (((lower_limit[index] is not None) and (
                            lower_extent[index] == lower_limit[index])) or (
                            diff_list[index] == 0)):
                        pass

                    else:
                        if upper_limit[index] is not None:
                            if upper_extent[index] + diff_list[index] >= upper_limit[index]:
                                new_upper_index = upper_limit[index]
                            else:
                                new_upper_index = upper_extent[index] + diff_list[index]
                        else:
                            new_upper_index = upper_extent[index] + diff_list[index]

                        new_upper[index] = new_upper_index

                        if lower_limit[index] is not None:
                            if lower_extent[index] - diff_list[index] <= lower_limit[index]:
                                new_lower_index = lower_limit[index]
                            else:
                                new_lower_index = lower_extent[index] - diff_list[index]
                        else:
                            new_lower_index = lower_extent[index] - diff_list[index]

                        new_lower[index] = new_lower_index

            for index in range(len(input_array)):
                upper_extent_constraint = input_array[index] <= new_upper[index]

                lower_extent_constraint = new_lower[index] <= input_array[index]

                solver.add(upper_extent_constraint,
                           lower_extent_constraint)

            within_current_constraint_upper_list = []
            within_current_constraint_lower_list = []
            for index in range(len(input_array)):
                within_current_constraint_upper_list.append(
                    input_array[index] <= upper_extent[index])
                within_current_constraint_lower_list.append(
                    lower_extent[index] <= input_array[index])

            constraint_upper = And(within_current_constraint_upper_list)
            constraint_lower = And(within_current_constraint_lower_list)
            constraint_ul = And(constraint_upper, constraint_lower)

            solver.add(Not(constraint_ul))

            result = check_solver(solver)

            if result == sat:

                search_count = search_count + 1

                _log_writer('#{0} {1} extended:'.format(
                    search_count, _SAMETIME),
                    [log_file, sys.stdout]
                )
                print_violation(
                    input_array, output_array, input_type,
                    model_array, model, solver, input_name)

                for index in range(len(input_array)):
                    upper_extent[index] = new_upper[index]
                    lower_extent[index] = new_lower[index]
                flag = True

            elif result == unknown:
                pass
            else:
                pass

            extend_volume = calc_volume(lower_extent, upper_extent)

            if system_time_out is not None:
                if time.time() > start_time + system_time_out:
                    # _log_writer('Volume: {0} / {1} = {2:%}'.format(
                    #     extend_volume, base_volume,
                    #     extend_volume / base_volume),
                    #     [sys.stdout, log_file])
                    solver.pop()
                    _log_writer('Time out find_extents_rev',
                                [log_file, sys.stdout])
                    return (lower_extent, upper_extent,
                            no_solution_spaces_fixed, True, diff_list)

            if search_time_out is not None:
                if time.time() > find_start + search_time_out:
                    # _log_writer('Volume: {0} / {1} = {2:%}'.format(
                    #     extend_volume, base_volume,
                    #     extend_volume / base_volume),
                    #     [sys.stdout, log_file])
                    solver.pop()
                    _log_writer('Time out find_extents_rev',
                                [log_file, sys.stdout])

                    return (lower_extent, upper_extent,
                            no_solution_spaces_fixed, False, diff_list)

            solver.pop()

        elif extend_arg == _SEPARATE:
            for index in range(len(input_array)):
                solver.push()

                if (cont_value_flag[index]) and (
                        input_array_solution[index] is not None) and (
                        input_used_or_not[
                            index] is True):

                    diff = diff_list[index]

                    for j in range(len(input_array)):
                        if (j != index) and (input_array_solution[j] is not None):
                            solver.add(input_array[j] <= upper_extent[j])
                            solver.add(lower_extent[j] <= input_array[j])

                    solver.push()

                    pre_upper_extend = upper_extent[index]
                    if ((upper_limit[index] is not None) and (
                            upper_extent[index] == upper_limit[index])) or (
                            diff == 0):
                        pass

                    else:
                        if upper_limit[index] is not None:
                            if upper_extent[index] + diff >= upper_limit[index]:
                                new_upper_index = upper_limit[index]
                            else:
                                new_upper_index = upper_extent[index] + diff
                        else:
                            new_upper_index = upper_extent[index] + diff

                        upper_extent_constraint1 = input_array[index] <= new_upper_index

                        upper_extent_constraint2 = upper_extent[index] < input_array[index]

                        solver.add(upper_extent_constraint1,
                                   upper_extent_constraint2)

                        result = check_solver(solver)

                        if result == sat:
                            search_count = search_count + 1
                            _log_writer('#{0} {1} extended: {2}'.format(
                                search_count, _UPPER, input_name[index]),
                                [log_file, sys.stdout]
                            )
                            print_violation(
                                input_array, output_array, input_type,
                                model_array, model, solver, input_name)

                            upper_extent[index] = new_upper_index
                            flag = True

                            d_key = str(index) + 'u'
                            if d_key in no_solution_space_tmp:
                                info = (index, 'u', no_solution_space_tmp.pop(
                                    d_key))
                                no_solution_spaces_fixed.append(info)
                            else:
                                pass

                        elif result == unsat:
                            d_key = str(index) + 'u'
                            no_sol_lower = []
                            no_sol_lower_eqsign = []
                            no_sol_upper = []
                            no_sol_upper_eqsign = []
                            for j in range(len(input_array)):
                                if input_array_solution[j] is not None:
                                    if j != index:
                                        no_sol_lower.append(lower_extent[j])
                                        no_sol_lower_eqsign.append('with_equal')
                                        no_sol_upper.append(upper_extent[j])
                                        no_sol_upper_eqsign.append('with_equal')
                                    else:
                                        no_sol_lower.append(upper_extent[index])
                                        no_sol_lower_eqsign.append(
                                            'without_equal')
                                        no_sol_upper.append(
                                            upper_extent[index] + diff)
                                        no_sol_upper_eqsign.append('with_equal')
                                else:
                                    no_sol_lower.append(None)
                                    no_sol_lower_eqsign.append(None)
                                    no_sol_upper.append(None)
                                    no_sol_upper_eqsign.append(None)

                            no_sol_tuple = (
                                tuple(no_sol_lower), tuple(no_sol_lower_eqsign),
                                tuple(no_sol_upper), tuple(no_sol_upper_eqsign))
                            no_solution_space_tmp[d_key] = no_sol_tuple

                    solver.pop()
                    solver.push()

                    pre_lower_extent = lower_extent[index]
                    if ((lower_limit[index] is not None) and (
                            lower_extent[index] == lower_limit[index])) or (
                            diff == 0):
                        pass

                    else:
                        if lower_limit[index] is not None:
                            if lower_extent[index] - diff <= lower_limit[index]:
                                new_lower_index = lower_limit[index]
                            else:
                                new_lower_index = lower_extent[index] - diff
                        else:
                            new_lower_index = lower_extent[index] - diff

                        lower_extent_constraint1 = new_lower_index <= input_array[index]

                        lower_extent_constraint2 = input_array[index] < lower_extent[index]

                        solver.add(lower_extent_constraint1,
                                   lower_extent_constraint2)

                        result = check_solver(solver)

                        if result == sat:
                            search_count = search_count + 1

                            _log_writer('#{0} {1} extended: {2}'.format(
                                search_count, _LOWER, input_name[index]),
                                [log_file, sys.stdout]
                            )
                            print_violation(
                                input_array, output_array, input_type,
                                model_array, model, solver, input_name)

                            lower_extent[index] = new_lower_index
                            flag = True

                            d_key = str(index) + 'l'
                            if d_key in no_solution_space_tmp:
                                info = (index, 'l', no_solution_space_tmp.pop(
                                    d_key))
                                no_solution_spaces_fixed.append(info)
                            else:
                                pass

                        elif result == unsat:
                            d_key = str(index) + 'l'
                            no_sol_lower = []
                            no_sol_lower_eqsign = []
                            no_sol_upper = []
                            no_sol_upper_eqsign = []
                            for j in range(len(input_array)):
                                if input_array_solution[j] is not None:
                                    if j != index:
                                        no_sol_lower.append(lower_extent[j])
                                        no_sol_lower_eqsign.append('with_equal')
                                        no_sol_upper.append(upper_extent[j])
                                        no_sol_upper_eqsign.append('with_equal')
                                    else:
                                        no_sol_lower.append(
                                            lower_extent[index] - diff)
                                        no_sol_lower_eqsign.append('with_equal')
                                        no_sol_upper.append(lower_extent[index])
                                        no_sol_upper_eqsign.append(
                                            'without_equal')
                                else:
                                    no_sol_lower.append(None)
                                    no_sol_lower_eqsign.append(None)
                                    no_sol_upper.append(None)
                                    no_sol_upper_eqsign.append(None)

                            no_sol_tuple = (
                                tuple(no_sol_lower), tuple(no_sol_lower_eqsign),
                                tuple(no_sol_upper), tuple(no_sol_upper_eqsign))
                            no_solution_space_tmp[d_key] = no_sol_tuple

                    solver.pop()

                solver.pop()

                extend_volume = calc_volume(lower_extent, upper_extent)

                if system_time_out is not None:
                    if time.time() > start_time + system_time_out:
                        # _log_writer('Volume: {0} / {1} = {2:%}'.format(
                        #     extend_volume, base_volume,
                        #     extend_volume / base_volume),
                        #     [sys.stdout, log_file])
                        solver.pop()
                        _log_writer('Time out find_extents_rev',
                                    [log_file, sys.stdout])
                        return (lower_extent, upper_extent,
                                no_solution_spaces_fixed, True, diff_list)

                if search_time_out is not None:
                    if time.time() > find_start + search_time_out:
                        # _log_writer('Volume: {0} / {1} = {2:%}'.format(
                        #     extend_volume, base_volume,
                        #     extend_volume / base_volume),
                        #     [sys.stdout, log_file])
                        solver.pop()
                        _log_writer('Time out find_extents_rev',
                                    [log_file, sys.stdout])

                        return (lower_extent, upper_extent,
                                no_solution_spaces_fixed, False, diff_list)

    solver.pop()

    # _log_writer('Volume: {0} / {1} = {2:%}'.format(
    #     extend_volume, base_volume, extend_volume / base_volume),
    #     [sys.stdout, log_file])

    return (
        lower_extent, upper_extent, no_solution_spaces_fixed, False, diff_list)


def check_splitting(space_tobe_split, lower_limit, upper_limit, vol_ratio):
    space_tobe_split_lower = space_tobe_split[0]
    space_tobe_split_upper = space_tobe_split[2]

    whole_vol = calc_volume(lower_limit, upper_limit)
    violation_vol = calc_volume(space_tobe_split_lower, space_tobe_split_upper)

    if (whole_vol * (vol_ratio / 100)) <= violation_vol:
        return True
    else:
        return False


def check_no_sol_space(space_tobe_split, no_solution_space):
    space_tobe_split_lower = space_tobe_split[0]
    space_tobe_split_lower_eqsign = space_tobe_split[1]
    space_tobe_split_upper = space_tobe_split[2]
    space_tobe_split_upper_eqsign = space_tobe_split[3]

    no_sol_lower = no_solution_space[2][0]
    no_sol_lower_eqsign = no_solution_space[2][1]
    no_sol_upper = no_solution_space[2][2]
    no_sol_upper_eqsign = no_solution_space[2][3]
    len_f = len(space_tobe_split_lower)

    for index in range(len_f):
        if space_tobe_split_lower_eqsign == 'without_equal' and \
                no_sol_lower_eqsign == 'with_equal':
            if space_tobe_split_lower[index] < no_sol_lower[index]:
                pass
            else:
                print('no_solution_space is not contained '
                      'in the space to be split')
                return False
        else:
            if space_tobe_split_lower[index] <= no_sol_lower[index]:
                pass
            else:
                print('no_solution_space is not contained '
                      'in the space to be split')
                return False

        if no_sol_upper_eqsign == 'with_equal' and \
                space_tobe_split_upper_eqsign == 'without_equal':
            if no_sol_upper[index] < space_tobe_split_upper[index]:
                pass
            else:
                print('no_solution_space is not contained '
                      'in the space to be split')
                return False
        else:
            if no_sol_upper[index] <= space_tobe_split_upper[index]:
                pass
            else:
                print('no_solution_space is not contained '
                      'in the space to be split')
                print(index)
                return False

    return True


def evaluate_splitting(split_spaces, split_space_inside, solver, input_array):

    split_spaces_list = split_spaces + [split_space_inside]
    satisfiability_list = []

    vol_sum = 0

    for split_space in split_spaces_list:
        if split_space is not None:
            lower_limit = split_space[0]
            lower_eqsign = split_space[1]
            upper_limit = split_space[2]
            upper_eqsign = split_space[3]

            solver.push()

            for f_index in range(len(lower_limit)):

                upper = upper_limit[f_index]
                if upper_eqsign[f_index] == 'with_equal':
                    upper_constraint = input_array[f_index] <= upper
                elif upper_eqsign[f_index] == 'without_equal':
                    upper_constraint = input_array[f_index] < upper
                else:
                    print('Error: an unexpected eqsign identifier is used')
                    assert False

                solver.add(upper_constraint)

                lower = lower_limit[f_index]
                if lower_eqsign[f_index] == 'with_equal':
                    lower_constraint = input_array[f_index] >= lower
                elif lower_eqsign[f_index] == 'without_equal':
                    lower_constraint = input_array[f_index] > lower
                else:
                    print('Error: an unexpected eqsign identifier is used')
                    assert False

                solver.add(lower_constraint)

            sat_unsat = check_solver(solver)
            satisfiability_list.append(sat_unsat)

            solver.pop()

            if sat_unsat == sat:
                vol = calc_volume(lower_limit, upper_limit)
                vol_sum += vol

        else:
            satisfiability_list.append(unsat)

    return vol_sum, satisfiability_list


def split_one(space_tobe_split, no_solution_space,
              solver, permutation_list_max, input_array):
    """

    :param space_tobe_split:
    :param no_solution_space:
    :param solver:
    :param permutation_list_max:
    :param input_array:
    :return:
    best_split_spaces splitした周りの空間
    best_split_space_inside：拡張した領域の起点を含む領域
    best_satisfiability_list：split_spacesの違反判定結果リスト

    """
    space_tobe_split_lower = space_tobe_split[0]
    space_tobe_split_lower_eqsign = space_tobe_split[1]
    space_tobe_split_upper = space_tobe_split[2]
    space_tobe_split_upper_eqsign = space_tobe_split[3]

    target_f_index = no_solution_space[0]
    upper_or_lower = no_solution_space[1]
    no_sol_lower = no_solution_space[2][0]
    no_sol_lower_eqsign = no_solution_space[2][1]
    no_sol_upper = no_solution_space[2][2]
    no_sol_upper_eqsign = no_solution_space[2][3]
    len_f = len(space_tobe_split_lower)

    permutation = []

    permutation_list = []

    for index in range(len_f):
        permutation.append((index, 'l'))
        permutation.append((index, 'u'))

    ran_seed = 0
    while len(permutation_list) < permutation_list_max and \
            len(permutation_list) < math.factorial(2 * len_f):

        random.seed(ran_seed)
        ran_seed += 1
        shuffle = random.sample(permutation, len(permutation))

        if shuffle not in permutation_list:
            permutation_list.append(shuffle)

    score_list = []
    split_result_list = []

    for num1, permutation in enumerate(permutation_list):
        split_spaces = []
        split_space_inside = None

        for num2, (index, ul) in enumerate(permutation):
            if ul == 'l':
                outer1_lower = space_tobe_split_lower[index]
                outer1_lower_eqsign = space_tobe_split_lower_eqsign[index]
                outer1_upper = no_sol_lower[index]

                if no_sol_lower_eqsign[index] == 'with_equal':
                    outer1_upper_eqsign = 'without_equal'
                elif no_sol_lower_eqsign[index] == 'without_equal':
                    outer1_upper_eqsign = 'with_equal'
                else:
                    print('Error: an unexpected eqsign identifier is used')
                    assert False

                if (outer1_lower != outer1_upper) or \
                        (outer1_lower_eqsign == 'with_equal' and
                         outer1_upper_eqsign == 'with_equal'):

                    new_space1_lower = list(space_tobe_split_lower)
                    new_space1_lower_eqsign = list(
                        space_tobe_split_lower_eqsign)
                    new_space1_upper = list(space_tobe_split_upper)
                    new_space1_upper_eqsign = list(
                        space_tobe_split_upper_eqsign)

                    new_space1_lower[index] = outer1_lower
                    new_space1_lower_eqsign[index] = outer1_lower_eqsign
                    new_space1_upper[index] = outer1_upper
                    new_space1_upper_eqsign[index] = outer1_upper_eqsign

                    for (i2, ul2) in permutation[:num2]:
                        if i2 != index:
                            if ul2 == 'l':
                                new_space1_lower[i2] = no_sol_lower[i2]
                                new_space1_lower_eqsign[i2] = no_sol_lower_eqsign[i2]
                            elif ul2 == 'u':
                                new_space1_upper[i2] = no_sol_upper[i2]
                                new_space1_upper_eqsign[i2] = no_sol_upper_eqsign[i2]
                            else:
                                print("ERROR: ul has an unexpected value")
                                assert False

                    if target_f_index == index and upper_or_lower == 'u':
                        split_space_inside = (
                            new_space1_lower, new_space1_lower_eqsign,
                            new_space1_upper, new_space1_upper_eqsign)
                    else:
                        split_spaces.append(
                            (new_space1_lower, new_space1_lower_eqsign,
                             new_space1_upper, new_space1_upper_eqsign))

                else:
                    split_spaces.append(None)

            elif ul == 'u':
                outer2_lower = no_sol_upper[index]
                if no_sol_upper_eqsign[index] == 'with_equal':
                    outer2_lower_eqsign = 'without_equal'
                elif no_sol_upper_eqsign[index] == 'without_equal':
                    outer2_lower_eqsign = 'with_equal'
                else:
                    print('Error: an unexpected eqsign identifier is used')
                    assert False
                outer2_upper = space_tobe_split_upper[index]
                outer2_upper_eqsign = space_tobe_split_upper_eqsign[index]

                if (outer2_lower != outer2_upper) or \
                        (outer2_lower_eqsign == 'with_equal' and
                         outer2_upper_eqsign == 'with_equal'):

                    new_space2_lower = list(space_tobe_split_lower)
                    new_space2_lower_eqsign = list(
                        space_tobe_split_lower_eqsign)
                    new_space2_upper = list(space_tobe_split_upper)
                    new_space2_upper_eqsign = list(
                        space_tobe_split_upper_eqsign)

                    new_space2_lower[index] = outer2_lower
                    new_space2_lower_eqsign[index] = outer2_lower_eqsign
                    new_space2_upper[index] = outer2_upper
                    new_space2_upper_eqsign[index] = outer2_upper_eqsign

                    for (i2, ul2) in permutation[:num2]:
                        if i2 != index:
                            if ul2 == 'l':
                                new_space2_lower[i2] = no_sol_lower[i2]
                                new_space2_lower_eqsign[i2] = no_sol_lower_eqsign[i2]
                            elif ul2 == 'u':
                                new_space2_upper[i2] = no_sol_upper[i2]
                                new_space2_upper_eqsign[i2] = no_sol_upper_eqsign[i2]
                            else:
                                print("ERROR: ul has an unexpected value")
                                assert False

                    if target_f_index == index and upper_or_lower == 'l':
                        split_space_inside = (
                            new_space2_lower, new_space2_lower_eqsign,
                            new_space2_upper, new_space2_upper_eqsign)
                    else:
                        split_spaces.append(
                            (new_space2_lower, new_space2_lower_eqsign,
                             new_space2_upper, new_space2_upper_eqsign))

                else:
                    split_spaces.append(None)

            else:
                print("ERROR: ul has an unexpected value")
                assert False

        # finish splitting for one permutation
        score, satisfiability_list = evaluate_splitting(
            split_spaces, split_space_inside, solver, input_array)

        score_list.append(score)
        split_result_list.append(
            (split_spaces, split_space_inside, satisfiability_list))

    (best_split_spaces,
     best_split_space_inside,
     best_satisfiability_list) = split_result_list[
        score_list.index(min(score_list))]

    return best_split_spaces, best_split_space_inside, best_satisfiability_list


def split_violation_space(space, lower_limit_tmp, upper_limit_tmp,
                          solver, vol_ratio, permutation_list_max, input_array):

    lower_eqsign = ['with_equal'] * len(space[0])
    upper_eqsign = ['with_equal'] * len(space[1])

    space_tobe_split = (space[0], lower_eqsign, space[1], upper_eqsign)
    no_solution_spaces_fixed = space[2]

    split_violation_spaces = []
    no_violation_spaces = []
    loop_index = 1

    while check_splitting(
            space_tobe_split, lower_limit_tmp, upper_limit_tmp, vol_ratio) and \
            loop_index <= len(no_solution_spaces_fixed):

        no_solution_space = no_solution_spaces_fixed[-loop_index]

        if check_no_sol_space(space_tobe_split, no_solution_space):
            pass
        else:
            loop_index += 1
            continue

        (split_spaces,
         split_space_inside,
         satisfiability_list) = split_one(
            space_tobe_split, no_solution_space,
            solver, permutation_list_max, input_array)

        split_spaces_sat = []
        split_spaces_unsat = []
        for index, sat_unsat in enumerate(satisfiability_list[:-1]):
            if sat_unsat == sat:
                split_spaces_sat.append(split_spaces[index])
            else:
                split_spaces_unsat.append(split_spaces[index])

        split_violation_spaces.extend(split_spaces_sat)
        no_violation_spaces.extend(split_spaces_unsat)

        if split_space_inside is not None:
            space_tobe_split = split_space_inside
        else:
            space_tobe_split = None
            break

        loop_index += 1

    if space_tobe_split is not None:
        split_violation_spaces.append(space_tobe_split)

    return split_violation_spaces, no_violation_spaces


def calc_volume(lower_limit, upper_limit):
    vol = 1
    for index in range(len(lower_limit)):
        len_of_side = abs(upper_limit[index] - lower_limit[index])
        vol = vol * len_of_side

    return vol


def print_violation(input_array, output_array, input_type,
                    model_array, model, solver, input_name):
    solution = solver.model()

    # Get z3 input
    z3_input = []
    for index in range(len(input_array)):

        if solution[input_array[index]] is None:
            pass
        else:

            if 'int' in str(input_type[index]):
                value = int(solution[input_array[index]].as_long())
                z3_input.append(value)

            elif 'float' in str(input_type[index]):
                value = float(solution[input_array[index]].as_fraction())
                z3_input.append(value)

            else:
                print("Error: input_type is not defined")
                assert False

    # Get z3 output
    z3_output = []
    for index in range(len(output_array)):
        obj = float(solution[output_array[index]].as_fraction())
        z3_output.append(obj)

    # Create xgboost input
    input_data_list = []
    for index in range(len(input_array)):

        if solution[input_array[index]] is None:
            input_data_list.append(0)
        else:
            if 'int' in str(input_type[index]):
                value = int(solution[input_array[index]].as_long())
                input_data_list.append(value)

            elif 'float' in str(input_type[index]):
                value = float(solution[input_array[index]].as_fraction())
                input_data_list.append(value)

            else:
                print("Error: input_type is not defined")
                assert False

    pd.options.display.max_columns = None
    pd.options.display.width = 3500
    pd.options.display.precision = 12

    if len(model_array) == 1:
        # predict
        df_input_nd = pd.DataFrame([input_data_list], columns=input_name)
        predict_output = model.predict(df_input_nd)
    else:
        df_input_nd = None
        predict_output = None

    # z3
    _log_writer(_LOG_MSG_INPUT_VALUE.format(_Z3, z3_input), log_file)
    _log_writer(_LOG_MSG_OUTPUT_VALUE.format(_Z3, z3_output), log_file)

    # xgboost
    _log_writer(_LOG_MSG_INPUT_VALUE.format(
        _XGBOOST, '\n' + str(df_input_nd)), log_file)
    _log_writer(_LOG_MSG_OUTPUT_VALUE.format(
        _XGBOOST, predict_output), log_file)


def _check_keys(keys, data):

    for key in keys:
        if key not in data:
            raise KeyError('"{}" not found'.format(key))


def _get_key_values(key, d_list, val_types):
    result = []
    for data in d_list:
        if key in data:
            if not any([isinstance(data[key], t) for t in val_types]):
                raise TypeError('"{0}" type is {1}'.format(
                    key, [t.__name__ for t in val_types]))
            result.append(data[key])

        else:
            result.append(None)

    return result


def _load_data_list(data_list_path):
    with open(data_list_path, 'r') as rs:
        data_list = json.load(rs)

    # Check data_list format
    if not isinstance(data_list, dict):
        raise TypeError('data list type is dict')

    _check_keys([_INPUT, _OUTPUT], data_list)

    input_data = data_list[_INPUT]
    output_data = data_list[_OUTPUT]

    # Check config format
    if not isinstance(input_data, list):
        raise TypeError('"{}" type is list'.format(_INPUT))

    if not isinstance(output_data, list):
        raise TypeError('"{}" type is list'.format(_OUTPUT))

    for in_d in input_data:
        _check_keys([_NAME, _CONT_VALUE_FLAG, _TYPE], in_d)

    for out_d in output_data:
        _check_keys([_NAME], out_d)

    # Get names
    input_names = _get_key_values(_NAME, input_data, [str])
    output_names = _get_key_values(_NAME, output_data, [str])
    # Get cont_value_flags
    cont_value_flags = _get_key_values(_CONT_VALUE_FLAG, input_data, [bool])
    # Get types
    types = _get_key_values(_TYPE, input_data, [str])

    if not all(t == 'int' or t == 'float' for t in types):
        raise ValueError('"{}" is "int" or "float"'.format(_TYPE))

    # Get uppers
    uppers = _get_key_values(_UPPER, input_data, [int, float])
    # Get lowers
    lowers = _get_key_values(_LOWER, input_data, [int, float])

    for upper, lower in zip(uppers, lowers):
        if upper is not None and lower is not None:
            if upper < lower:
                raise ValueError(
                    '"{0}":{2} < "{1}":{3}'.format(
                        _UPPER, _LOWER, upper, lower
                    )
                )

    return input_names, output_names, cont_value_flags, types, uppers, lowers


def _is_empty_file(file_path):
    with open(file_path, 'r') as rs:
        line = rs.readline()

    return line == ''


def _get_conf_path_value(conf_dir, key, data_list):
    if key not in data_list:
        raise KeyError('"{}" not found'.format(key))

    elif not isinstance(data_list[key], str):
        raise TypeError('"{}" type str'.format(key))

    else:
        val = data_list[key]

    if not Path(val).is_absolute():
        val = conf_dir.joinpath(val).resolve()

    if not Path(val).exists():
        raise FileNotFoundError("{} is not found".format(val))

    elif _is_empty_file(val):
        raise ValueError('{} is empty'.format(val))

    return val


def _load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError("{} is not found".format(config_path))

    conf_dir = Path(config_path).absolute().parent

    with open(config_path, 'r') as rs:
        data_list = json.load(rs)

    # Check config format and get value
    if not isinstance(data_list, dict):
        raise TypeError('config type is dict')

    # Get search range ratio
    if _SEARCH_RANGE_RATIO not in data_list:
        search_range_ratio = 100

    elif not (
            isinstance(data_list[_SEARCH_RANGE_RATIO], int) or
            isinstance(data_list[_SEARCH_RANGE_RATIO], float)):
        raise TypeError('"{}" type is int or float'.format(_SEARCH_RANGE_RATIO))
    else:
        search_range_ratio = data_list[_SEARCH_RANGE_RATIO]

    if search_range_ratio <= 1:
        raise ValueError('"{}" must greater 1'.format(_SEARCH_RANGE_RATIO))

    # Get buffer range ratio
    if _BUFFER_RANGE_RATIO not in data_list:
        buff_range_ratio = None

    elif not (
            isinstance(data_list[_BUFFER_RANGE_RATIO], int) or
            isinstance(data_list[_BUFFER_RANGE_RATIO], float)):
        raise TypeError(
            '"{}" type is int or float'.format(_BUFFER_RANGE_RATIO))
    else:
        buff_range_ratio = data_list[_BUFFER_RANGE_RATIO]

    if buff_range_ratio == 0:
        buff_range_ratio = None

    # Get Property file path
    prop_path = _get_conf_path_value(conf_dir, _PROP_PATH, data_list)

    # Get Data list file path
    data_list_path = _get_conf_path_value(conf_dir, _DATA_LIST_PATH, data_list)

    # Get Condition file path
    if _COND_PATH not in data_list:
        cond_path = None

    else:
        cond_path = _get_conf_path_value(conf_dir, _COND_PATH, data_list)

    # Get System Time out
    if _SYSTEM_TIMEOUT not in data_list:
        system_time_out = None
    elif not (
            isinstance(data_list[_SYSTEM_TIMEOUT], int) or
            isinstance(data_list[_SYSTEM_TIMEOUT], float)):
        raise TypeError('"{}" type is int or float'.format(_SYSTEM_TIMEOUT))

    else:
        system_time_out = data_list[_SYSTEM_TIMEOUT]

    # Get Search Function Time out
    if _SEARCH_TIMEOUT not in data_list:
        search_time_out = None
    elif not (
            isinstance(data_list[_SEARCH_TIMEOUT], int) or
            isinstance(data_list[_SEARCH_TIMEOUT], float)):
        raise TypeError('"{}" type is int or float'.format(_SEARCH_TIMEOUT))

    else:
        search_time_out = data_list[_SEARCH_TIMEOUT]

    # Get Extend Arg
    if _EXTEND_ARG not in data_list:
        extend_arg = _SEPARATE

    elif data_list[_EXTEND_ARG] not in [_SEPARATE, _SAMETIME]:
        raise ValueError('"{0}" value is "{1}" or "{2}"'.format(
            _EXTEND_ARG, _SEPARATE, _SAMETIME))
    else:
        extend_arg = data_list[_EXTEND_ARG]

    # Get splitting switch
    if _SPLITTING not in data_list:
        splitting = False

    elif not isinstance(data_list[_SPLITTING], bool):
        raise TypeError('"{}" type is bool'.format(_SPLITTING))
    else:
        splitting = data_list[_SPLITTING]

    if splitting:
        # Get vol ratio
        if _VOL_RATIO not in data_list:
            vol_ratio = 100

        elif not (
                isinstance(data_list[_VOL_RATIO], int) or
                isinstance(data_list[_VOL_RATIO], float)):
            raise TypeError('"{}" type is int or float'.format(_VOL_RATIO))
        else:
            vol_ratio = data_list[_VOL_RATIO]

        # Get permutation list_max
        if _PERMUTATION_LIST_MAX not in data_list:
            permutation_list_max = 10

        elif not (
                isinstance(data_list[_PERMUTATION_LIST_MAX], int) or
                isinstance(data_list[_PERMUTATION_LIST_MAX], float)):
            raise TypeError('"{}" type is int or float'.format(_PERMUTATION_LIST_MAX))
        else:
            permutation_list_max = data_list[_PERMUTATION_LIST_MAX]
    else:
        vol_ratio = None
        permutation_list_max = None

    return (search_range_ratio, buff_range_ratio,
            data_list_path, prop_path, cond_path,
            system_time_out, search_time_out, extend_arg,
            splitting, vol_ratio, permutation_list_max)


def _log_writer(msg, out_stream=sys.stdout):
    if isinstance(out_stream, list):
        for out in out_stream:
            print(msg, file=out, flush=True)
    else:
        print(msg, file=out_stream, flush=True)


def print_extent(lower_extent, upper_extent,
                 input_name, input_used_or_not):
    _log_writer('Violation Range', [log_file, sys.stdout])

    _log_writer('Range:', [log_file, sys.stdout])
    for index in range(len(upper_extent)):
        if input_used_or_not[index]:
            _log_writer(
                '  {0} : {1} <= to <= {2}'.format(
                    input_name[index],
                    lower_extent[index], upper_extent[index]),
                [log_file, sys.stdout]
            )


def print_split_spaces(split_space):
    lower = split_space[0]
    lower_eqsign = split_space[1]
    upper = split_space[2]
    upper_eqsign = split_space[3]

    # _log_writer('===============Split Space===============', log_file)

    if split_space is not None:
        for index in range(len(lower)):
            _log_writer(
                'f{0} : {1} {2} to {3} {4}'.format(
                    index,
                    lower[index],
                    '<' if lower_eqsign[index] == 'without_equal' else '<=',
                    '<' if upper_eqsign[index] == 'without_equal' else '<=',
                    upper[index]), [sys.stdout, log_file])
    else:
        _log_writer('None', [sys.stdout, log_file])


def main(models, dataset=None, config_path=None):
    cur_dir = Path(__file__).absolute().parent

    start_time = time.time()

    time_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = cur_dir.joinpath("log", str(time_id + 'log'))
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name = str(log_dir.joinpath('log_file.txt'))

    try:
        global log_file
        log_file = open(log_file_name, 'a', encoding='utf-8')

        (search_range_ratio, buff_range_ratio,
         data_list_path, prop_path, cond_path,
         system_time_out, search_time_out, extend_arg,
         splitting, vol_ratio, permutation_list_max) = _load_config(config_path)

        (input_name, output_name, cont_value_flag,
         input_type, uppers, lowers) = _load_data_list(data_list_path)

        lower_limit, upper_limit = cast_upperlower(lowers, uppers, input_type)

        for index, upper in enumerate(upper_limit):
            lower = lower_limit[index]
            if upper is None:
                print(_LOG_MSG_WARN_UPPER_LIMIT.format(input_name[index]))

            if lower is None:
                print(_LOG_MSG_WARN_LOWER_LIMIT.format(input_name[index]))

        input_array = dec_exp_vars_rev(input_type)

        for index, name in enumerate(input_name):
            print('{0} : {1} : {2}'.format(input_array[index], name,
                                           input_type[index]))

        model_array = []
        model_name_array = models
        for model_name in model_name_array:
            with open(model_name, 'rb') as model_file:
                model_array.append(pickle.load(model_file))

        if len(model_array) != len(output_name):
            print("データ数不整合4", file=sys.stderr)
            assert False

        solver = Solver()

        output_array = []
        sub_c_array_array = []
        c_sum_array = []
        input_used_or_not = [False] * len(input_array)

        for model_index, model in enumerate(model_array):

            if False:
                model = model.best_estimator_
                booster = model.get_booster()

            else:
                booster = model.get_booster()

            tree_tmp_dir = cur_dir.joinpath('tree_tmp')
            if not tree_tmp_dir.exists():
                tree_tmp_dir.mkdir(parents=True, exist_ok=True)
            tree_name = str(tree_tmp_dir.joinpath(
                'tree_' + str(model_index) + '.json'))
            dump_model(booster, tree_name, dump_format="json")

            # for num in range(len(booster.get_dump())):
            #     graph1 = xgb.to_graphviz(model, num_trees=num)
            #     graph1.format = 'png'
            #     graph1.render('tree_' + str(num))

            with open(tree_name, 'r') as fjson:
                djson = json.load(fjson)

            for tree in djson:
                input_name_replacer(tree, input_name)

            sub_c_array = add_trees_consts_to_solver(djson, input_array,
                                                     input_type, solver,
                                                     input_used_or_not)
            sub_c_array_array.append(sub_c_array)

            c_sum = add_trees_relation_regressor(sub_c_array, solver)
            c_sum_array.append(c_sum)

            base_score = model.get_xgb_params()['base_score']

            output_array.append(Real(_OUTPUT + str(model_index)))
            solver.add(
                output_array[model_index] == (
                            c_sum_array[model_index] + base_score))

        for index, or_not in enumerate(input_used_or_not):
            if or_not is False:
                _log_writer(_LOG_MSG_WARN_VARIAVLES.format(input_name[index]),
                            [sys.stdout, log_file])

        add_property(input_array, output_array, input_name, output_name,
                     solver, prop_path)

        add_upperlower_constraints(input_array, solver,
                                   upper_limit, lower_limit)

        if cond_path:
            add_other_constraints(input_array, output_array,
                                  input_name, output_name,
                                  solver, cond_path)

        solver.push()

        violation_spaces_list = []
        violation_num = 0

        while True:
            print("\nVerification Starts")
            satisfiability = check_solver(solver)

            if satisfiability == sat:
                print('Violating input value exists')
                violation_num += 1

                _log_writer(
                    'Range Extraction Starts', [log_file, sys.stdout])
                print_violation(input_array, output_array, input_type,
                                model_array, model, solver, input_name)

                (lower_extent, upper_extent, no_solution_spaces_fixed,
                 time_out_flag, diff_list) = find_extents_rev(
                    solver, model, lower_limit, upper_limit,
                    input_array, output_array, input_type,
                    model_array, cont_value_flag,
                    search_range_ratio,
                    input_name, input_used_or_not,
                    start_time, system_time_out, search_time_out, extend_arg)

                print_extent(
                    lower_extent, upper_extent, input_name, input_used_or_not)

                if time_out_flag:
                    return

                buff_list = []

                if buff_range_ratio is not None:
                    for index in range(len(input_array)):
                        buff = calc_diff(lower_limit[index], upper_limit[index],
                                         input_type[index], buff_range_ratio)
                        buff_list.append(buff)

                    for index, buff in enumerate(buff_list):
                        lower_extent[index] -= buff
                        upper_extent[index] += buff

                violation_spaces_list.append(
                    (tuple(lower_extent),
                     tuple(upper_extent),
                     tuple(no_solution_spaces_fixed)))

                constraint_extent = And(And(
                    [input_array[i] <= upper_extent[i] for i in
                     range(len(input_array))]),
                    And([input_array[i] >= lower_extent[i] for i
                         in range(len(input_array))]))

                solver.add(
                    Not(constraint_extent))

            else:
                print('No violating input value exists')
                break

        solver.pop()

        _log_writer("\nThe number of the violations ranges is {0}\n".format(
            violation_num), [sys.stdout, log_file])

        if splitting and extend_arg == _SEPARATE:
            split_result_list = []

            print('\nSplitting Starts')

            if len(violation_spaces_list) > 0:

                _log_writer(
                    '===============Proposed method===============', log_file)
                for num, space in enumerate(violation_spaces_list):

                    print('Splitting {0}th violation range'.format(num))

                    (split_violation_spaces,
                     split_no_violation_spaces) = split_violation_space(
                        space, lower_limit, upper_limit, solver,
                        vol_ratio, permutation_list_max, input_array)

                    split_result = (
                        split_violation_spaces, split_no_violation_spaces)
                    split_result_list.append(split_result)

        solver.reset()

        elapsed_time = time.time() - start_time
        _log_writer('\n====================Result==============',
                    [sys.stdout, log_file])

        if splitting and extend_arg == _SEPARATE:
            logging_split_result(violation_spaces_list, split_result_list,
                                 input_name, input_used_or_not)

        _log_writer('Number of executions of SMT solver : {}'.format(
            num_call_solver[0]), [sys.stdout, log_file])
        _log_writer('SMT solver run time (s) : {}'.format(num_call_solver[1]),
                    [sys.stdout, log_file])
        _log_writer('Total run time (s) : {}'.format(elapsed_time),
                    [sys.stdout, log_file])

        _log_writer('\n-----------------------\n', [sys.stdout, log_file])

        return

    except Exception as e:
        sys.stdout.flush()
        traceback.print_exc()
        return

    finally:
        time.sleep(0.1)
        print("", flush=True)
        log_file.flush()
        log_file.close()


def logging_split_result(violation_spaces_list, split_result_list,
                         input_name, input_used_or_not):

    for num, (violation_space, split_result) in enumerate(
            zip(violation_spaces_list, split_result_list)):
        space_before_splitting_lower = violation_space[0]
        space_before_splitting_upper = violation_space[1]

        _log_writer(
            '\n===============Space to be split %d===============' % num,
            [log_file])
        print_extent(space_before_splitting_lower,
                     space_before_splitting_upper,
                     input_name, input_used_or_not)

        for i, elm in enumerate(split_result[0]):
            _log_writer(
                '\n===============Split Spaces %d===============' % i,
                [sys.stdout, log_file])
            print_split_spaces(elm)

        before_volume = calc_volume(space_before_splitting_lower,
                                    space_before_splitting_upper)
        _log_writer(
            '\n===============Before splitting Volume===============\n'
            '{}\n'.format(before_volume), [sys.stdout, log_file])

        split_volume_sum = 0
        for i, elm in enumerate(split_result[0]):
            split_volume = calc_volume(elm[0], elm[2])
            split_volume_sum = split_volume_sum + split_volume
            _log_writer(
                '===============After splitting Volume {0}===============\n'
                '{1}\n'.format(i, split_volume), [sys.stdout, log_file])

        _log_writer(
            '===============After splitting Volume Sum===============\n'
            '{}\n'.format(split_volume_sum), [sys.stdout, log_file])

        _log_writer(
            'After splitting Volume Sum / Before splitting Volume : {}\n'.format(
                split_volume_sum / before_volume, [sys.stdout, log_file]))
