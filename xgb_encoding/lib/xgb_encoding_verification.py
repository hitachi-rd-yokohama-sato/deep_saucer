import sys
import pickle
import pandas as pd
import json
import importlib.util as iu

from z3 import *
import copy
import time
import datetime
import math

import xgboost as xgb
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
_LOG_MSG_DEBUG_INPUT_VALUE = '  Debug : input to {0} is {1}'
_LOG_MSG_OUTPUT_VALUE = '  output from {0} is {1}'
_LOG_MSG_WARN_VARIAVLES = 'Warning: Explanatory variable "{}" ' \
                          'not used in model.'
_LOG_MSG_WARN_UPPER_LIMIT = 'Warning: Explanatory variable "{}" ' \
                            'was not set Upper limit.' \
                            'Search violation may not stop.'
_LOG_MSG_WARN_LOWER_LIMIT = 'Warning: Explanatory variable "{}" ' \
                            'was not set Lower limit.' \
                            'Search violation may not stop.'

_LOG_MSG_TIME_OUT = 'Time out find_extents_rev'

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
_MODE = 'mode'
_REGRESSOR = 'regressor'
_CLASSIFIER = 'classifier'
_CONV_PATH = 'conv_file_path'
_CONV_NAME = 'conv_func_name'


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

_CONVERSION = 'conversion'
_CONVERTED = 'converted'
_INPUT_CONVERTED = _INPUT + '_' + _CONVERTED
_OUTPUT_CONVERTED = _OUTPUT + '_' + _CONVERTED

num_call_solver = [0, 0.0]


def check_solver(solver):
    num_call_solver[0] += 1
    time_before_check = time.time()
    result = solver.check()
    num_call_solver[1] += (time.time() - time_before_check)

    return result


def dec_exp_vars_rev(input_type, **kwargs):

    conversion = kwargs.get(_CONVERSION, False)

    input_array = []
    for i in range(len(input_type)):

        if 'int' in str(input_type[i]):
            if not conversion:
                input_array.append(Int(_INPUT + str(i)))
            else:
                input_array.append(Int(_INPUT_CONVERTED + str(i)))
        elif 'float' in str(input_type[i]):
            if not conversion:
                input_array.append(Real(_INPUT + str(i)))
            else:
                input_array.append(Real(_INPUT_CONVERTED + str(i)))
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
            print('Error ' + name_var + ' could not be repleced',
                  file=sys.stderr)
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
            print('Error: the length of input_used_or_not is invalid',
                  file=sys.stderr)
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
              "the XGBoost model has an unexpected attribute", file=sys.stderr)
        assert False


def add_trees_consts_to_solver(djson, input_array, input_type, solver,
                               input_used_or_not, **kwargs):

    conversion = kwargs.get(_CONVERSION, False)
    sub_c_array = []
    for tree_id, tree in enumerate(djson):

        if not conversion:
            out_var = Real('c{0}'.format(tree_id))
        else:
            out_var = Real('c_{0}{1}'.format(_CONVERSION, tree_id))

        sub_c_array.append(out_var)
        antecedent = []
        add_constraints(input_array, input_type, out_var, tree, antecedent, solver,
                        input_used_or_not)
    return sub_c_array


def add_trees_relation_regressor(sub_c_array, solver, **kwargs):
    conversion = kwargs.get(_CONVERSION, False)
    if not conversion:
        c_sum = Real('c_sum')
    else:
        c_sum = Real('c_sum_' + _CONVERTED)
    added = sub_c_array[0]
    for index in range(1, len(sub_c_array)):
        added = added + sub_c_array[index]

    rel_formula = (c_sum == added)
    solver.add(rel_formula)

    return c_sum


def add_trees_relation_classifier(out_array, sub_c_array, category, base_score, solver):
    for index, out_var in enumerate(out_array):
        added = sub_c_array[index]
        while index + category < len(sub_c_array):
            index = index + category
            added = added + sub_c_array[index]

        added = added + base_score

        rel_formula = (out_var == added)
        solver.add(rel_formula)


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
                 input_array_converted, output_array_converted,
                 in_names, out_names, solver, constraint_file):

    prop_str = z3util.parse_constraint(constraint_file, in_names, out_names)
    neg_prop = Not(eval(prop_str))
    solver.add(neg_prop)

    print_solver = Solver()
    print_solver.add(neg_prop)
    _log_writer('===== property =====', log_file)
    _log_writer(print_solver.sexpr(), log_file)

    del print_solver


def _cast_values(value_list, type_list):
    cast_value_list = []
    for _value, _type in zip(value_list, type_list):
        if _value is not None:
            try:
                if 'int' in str(_type):
                    cast_value_list.append(int(_value))

                elif 'float' in str(_type):
                    cast_value_list.append(Fraction(_value))

                else:
                    print("Unexpected input_type is designated", file=sys.stderr)
                    assert False
            except:
                raise ValueError
        else:
            cast_value_list.append(None)

    return cast_value_list


def cast_upperlower(lower_limit, upper_limit, input_type):

    try:
        upper_limit_list = _cast_values(upper_limit, input_type)

    except ValueError:
        print("Unexpected value is given as upper_bound", file=sys.stderr)
        assert False

    try:
        lower_limit_list = _cast_values(lower_limit, input_type)

    except ValueError:
        print("Unexpected value is given as lower_bound", file=sys.stderr)
        assert False

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
        print("Error: input_type is not defined", file=sys.stderr)
        assert False

    return diff


def _check_timeout(conf_time_out, start_time):
    if conf_time_out is not None:
        if time.time() > start_time + conf_time_out:
            return True

    return False


def find_extents_rev(solver, model, lower_limit, upper_limit,
                     input_array, output_array,
                     input_array_converted, output_array_converted,
                     input_type, cont_value_flag,
                     input_name, input_used_or_not, start_time, conf):

    find_start = time.time()

    # base_volume = calc_volume(lower_limit, upper_limit, input_used_or_not)
    m = solver.model()
    input_array_solution = get_z3_solution(m, input_array, input_type)

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
    for lower, upper, _type in zip(lower_limit, upper_limit, input_type):
        diff = calc_diff(lower, upper, _type, conf.search_range_ratio)
        diff_list.append(diff)

    flag = True
    search_count = 0
    while flag is True:
        flag = False

        if conf.extend_arg == _SAMETIME:
            search_count, flag = _find_extents_rev_sametime(
                flag, lower_extent, upper_extent, diff_list,
                input_array_solution, search_count,
                solver, model, lower_limit, upper_limit,
                input_array, output_array,
                input_array_converted, output_array_converted,
                input_type, cont_value_flag,
                input_name, input_used_or_not, conf)

            # Check Time out
            if _check_timeout(conf.system_time_out, start_time):
                solver.pop()

                _log_writer(_LOG_MSG_TIME_OUT, [log_file, sys.stdout])

                return (lower_extent, upper_extent,
                        no_solution_spaces_fixed, True, diff_list)

            if _check_timeout(conf.search_time_out, find_start):
                solver.pop()

                _log_writer(_LOG_MSG_TIME_OUT, [log_file, sys.stdout])

                return (lower_extent, upper_extent,
                        no_solution_spaces_fixed, False, diff_list)

        elif conf.extend_arg == _SEPARATE:
            for index in range(len(input_array)):
                solver.push()

                if (cont_value_flag[index]) and (
                        input_array_solution[index] is not None) and (
                            input_used_or_not[index] is True):

                    diff = diff_list[index]

                    for _index in range(len(input_array)):
                        if (_index != index) and (
                                input_array_solution[_index] is not None):
                            solver.add(
                                input_array[_index] <= upper_extent[_index])
                            solver.add(
                                lower_extent[_index] <= input_array[_index])

                    search_count, flag = _find_extents_rev_upper(
                        index, flag, lower_extent, upper_extent, diff,
                        input_array_solution, search_count,
                        no_solution_space_tmp, no_solution_spaces_fixed,
                        solver, model, upper_limit,
                        input_array, output_array,
                        input_array_converted, output_array_converted,
                        input_type, input_name, conf)

                    search_count, flag = _find_extents_rev_lower(
                        index, flag, lower_extent, upper_extent, diff,
                        input_array_solution, search_count,
                        no_solution_space_tmp, no_solution_spaces_fixed,
                        solver, model, lower_limit,
                        input_array, output_array,
                        input_array_converted, output_array_converted,
                        input_type, input_name, conf)

                solver.pop()

                # Check Time out
                if _check_timeout(conf.system_time_out, start_time):
                    solver.pop()

                    _log_writer(_LOG_MSG_TIME_OUT, [log_file, sys.stdout])

                    return (lower_extent, upper_extent,
                            no_solution_spaces_fixed, True, diff_list)

                if _check_timeout(conf.search_time_out, find_start):
                    solver.pop()

                    _log_writer(_LOG_MSG_TIME_OUT, [log_file, sys.stdout])

                    return (lower_extent, upper_extent,
                            no_solution_spaces_fixed, False, diff_list)

    solver.pop()

    return (
        lower_extent, upper_extent, no_solution_spaces_fixed, False, diff_list)


def _find_extents_rev_sametime(flag, lower_extent, upper_extent, diff_list,
                               input_array_solution, search_count,
                               solver, model, lower_limit, upper_limit,
                               input_array, output_array,
                               input_array_converted, output_array_converted,
                               input_type, cont_value_flag,
                               input_name, input_used_or_not, conf):
    solver.push()

    new_upper = copy.deepcopy(upper_extent)
    new_lower = copy.deepcopy(lower_extent)
    for index in range(len(input_array)):
        if (cont_value_flag[index]) and (
                input_array_solution[index] is not None) and (
                    input_used_or_not[index] is True):

            diff = diff_list[index]

            if (((upper_limit[index] is not None) and
                 (upper_extent[index] == upper_limit[index])) or (diff == 0)) and (
                    ((lower_limit[index] is not None) and
                     (lower_extent[index] == lower_limit[index])) or (diff == 0)):
                pass

            else:
                if upper_limit[index] is not None:
                    if upper_extent[index] + diff >= upper_limit[index]:
                        new_upper_extent = upper_limit[index]
                    else:
                        new_upper_extent = upper_extent[index] + diff
                else:
                    new_upper_extent = upper_extent[index] + diff

                new_upper[index] = new_upper_extent

                if lower_limit[index] is not None:
                    if lower_extent[index] - diff <= lower_limit[index]:
                        new_lower_extent = lower_limit[index]
                    else:
                        new_lower_extent = lower_extent[index] - diff
                else:
                    new_lower_extent = lower_extent[index] - diff

                new_lower[index] = new_lower_extent

    for index in range(len(input_array)):
        upper_extent_constraint = input_array[index] <= new_upper[index]

        lower_extent_constraint = new_lower[index] <= input_array[index]

        solver.add(upper_extent_constraint, lower_extent_constraint)

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
        print_extent(new_lower, new_upper, input_name, input_used_or_not)

        print_violation(
            input_array, output_array,
            input_array_converted, output_array_converted, input_type,
            model, solver, input_name, conf)

        for index in range(len(input_array)):
            upper_extent[index] = new_upper[index]
            lower_extent[index] = new_lower[index]

        flag = True

    elif result == unknown:
        pass
    else:
        pass

    solver.pop()

    return search_count, flag


def _find_extents_rev_upper(index, flag, lower_extent, upper_extent, diff,
                            input_array_solution, search_count,
                            no_solution_space_tmp, no_solution_spaces_fixed,
                            solver, model, upper_limit,
                            input_array, output_array,
                            input_array_converted, output_array_converted,
                            input_type, input_name, conf):
    solver.push()

    pre_upper_extend = upper_extent[index]
    if ((upper_limit[index] is not None) and (
            upper_extent[index] == upper_limit[index])) or (diff == 0):
        pass

    else:
        if upper_limit[index] is not None:
            if upper_extent[index] + diff >= upper_limit[index]:
                new_upper_extent = upper_limit[index]
            else:
                new_upper_extent = upper_extent[index] + diff
        else:
            new_upper_extent = upper_extent[index] + diff

        upper_extent_constraint1 = input_array[index] <= new_upper_extent

        upper_extent_constraint2 = upper_extent[index] < input_array[index]

        solver.add(upper_extent_constraint1, upper_extent_constraint2)

        result = check_solver(solver)

        if result == sat:
            search_count = search_count + 1

            if 'int' in str(input_type[index]):
                _log_writer(
                    '#{0} {1} extended : {2} : {3} -> {4}'.format(
                        search_count, _UPPER,
                        input_name[index], upper_extent[index],
                        new_upper_extent),
                    [log_file, sys.stdout])

            elif 'float' in str(input_type[index]):
                _log_writer(
                    '#{0} {1} extended : {2} : {3} -> {4}'.format(
                        search_count, _UPPER,
                        input_name[index],
                        float(upper_extent[index]),
                        float(new_upper_extent)),
                    [log_file, sys.stdout])

            print_violation(
                input_array, output_array,
                input_array_converted, output_array_converted,
                input_type, model, solver, input_name, conf)

            upper_extent[index] = new_upper_extent
            flag = True

            d_key = str(index) + 'u'
            if d_key in no_solution_space_tmp:
                info = (index, 'u', no_solution_space_tmp.pop(d_key))
                no_solution_spaces_fixed.append(info)
            else:
                pass

        elif result == unsat:
            d_key = str(index) + 'u'
            no_sol_lower = []
            no_sol_lower_eqsign = []
            no_sol_upper = []
            no_sol_upper_eqsign = []
            for _index in range(len(input_array)):
                if input_array_solution[_index] is not None:
                    if _index != index:
                        no_sol_lower.append(lower_extent[_index])
                        no_sol_lower_eqsign.append('with_equal')
                        no_sol_upper.append(upper_extent[_index])
                        no_sol_upper_eqsign.append('with_equal')
                    else:
                        no_sol_lower.append(upper_extent[index])
                        no_sol_lower_eqsign.append('without_equal')
                        no_sol_upper.append(upper_extent[index] + diff)
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

    return search_count, flag


def _find_extents_rev_lower(index, flag, lower_extent, upper_extent, diff,
                            input_array_solution, search_count,
                            no_solution_space_tmp, no_solution_spaces_fixed,
                            solver, model, lower_limit,
                            input_array, output_array,
                            input_array_converted, output_array_converted,
                            input_type, input_name, conf):

    solver.push()

    pre_lower_extent = lower_extent[index]
    if ((lower_limit[index] is not None) and (
            lower_extent[index] == lower_limit[index])) or (diff == 0):
        pass

    else:
        if lower_limit[index] is not None:
            if lower_extent[index] - diff <= lower_limit[index]:
                new_lower_extent = lower_limit[index]
            else:
                new_lower_extent = lower_extent[index] - diff
        else:
            new_lower_extent = lower_extent[index] - diff

        lower_extent_constraint1 = new_lower_extent <= input_array[index]

        lower_extent_constraint2 = input_array[index] < lower_extent[index]

        solver.add(lower_extent_constraint1, lower_extent_constraint2)

        result = check_solver(solver)

        if result == sat:
            search_count = search_count + 1

            if 'int' in str(input_type[index]):
                _log_writer(
                    '#{0} {1} extended: {2} : {3} -> {4}'.format(
                        search_count, _LOWER,
                        input_name[index], lower_extent[index],
                        new_lower_extent),
                    [log_file, sys.stdout])

            elif 'float' in str(input_type[index]):
                _log_writer(
                    '#{0} {1} extended: {2} : {3} -> {4}'.format(
                        search_count, _LOWER,
                        input_name[index],
                        float(lower_extent[index]),
                        float(new_lower_extent)),
                    [log_file, sys.stdout])

            print_violation(
                input_array, output_array,
                input_array_converted, output_array_converted,
                input_type, model, solver, input_name, conf)

            lower_extent[index] = new_lower_extent
            flag = True

            d_key = str(index) + 'l'
            if d_key in no_solution_space_tmp:
                info = (index, 'l', no_solution_space_tmp.pop(d_key))
                no_solution_spaces_fixed.append(info)
            else:
                pass

        elif result == unsat:
            d_key = str(index) + 'l'
            no_sol_lower = []
            no_sol_lower_eqsign = []
            no_sol_upper = []
            no_sol_upper_eqsign = []
            for _index in range(len(input_array)):
                if input_array_solution[_index] is not None:
                    if _index != index:
                        no_sol_lower.append(lower_extent[_index])
                        no_sol_lower_eqsign.append('with_equal')
                        no_sol_upper.append(upper_extent[_index])
                        no_sol_upper_eqsign.append('with_equal')
                    else:
                        no_sol_lower.append(lower_extent[index] - diff)
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

    return search_count, flag


def check_splitting(space_tobe_split, lower_limit, upper_limit, vol_ratio, input_used_or_not):
    space_tobe_split_lower = space_tobe_split[0]
    space_tobe_split_upper = space_tobe_split[2]

    whole_vol = calc_volume(lower_limit, upper_limit, input_used_or_not)
    violation_vol = calc_volume(
        space_tobe_split_lower, space_tobe_split_upper, input_used_or_not)

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


def evaluate_splitting(split_spaces, split_space_inside, solver,
                       input_array, input_used_or_not):

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
                    print('Error: an unexpected eqsign identifier is used',
                          file=sys.stderr)
                    assert False

                solver.add(upper_constraint)

                lower = lower_limit[f_index]
                if lower_eqsign[f_index] == 'with_equal':
                    lower_constraint = input_array[f_index] >= lower
                elif lower_eqsign[f_index] == 'without_equal':
                    lower_constraint = input_array[f_index] > lower
                else:
                    print('Error: an unexpected eqsign identifier is used',
                          file=sys.stderr)
                    assert False

                solver.add(lower_constraint)

            sat_unsat = check_solver(solver)
            satisfiability_list.append(sat_unsat)

            solver.pop()

            if sat_unsat == sat:
                vol = calc_volume(lower_limit, upper_limit, input_used_or_not)
                vol_sum += vol

        else:
            satisfiability_list.append(unsat)

    return vol_sum, satisfiability_list


def split_one(space_tobe_split, no_solution_space,
              solver, permutation_list_max,
              input_array, input_used_or_not):
    """

    Args:
        space_tobe_split:
        no_solution_space:
        solver:
        permutation_list_max:
        input_array:
        input_used_or_not:

    Returns:
        tuple: split informations.

        best_split_spaces: Space around split.
        best_split_space_inside: A region that contains the origin of the expanded region.
        best_satisfiability_list: Violation judgment result list of split_spaces.
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

    for _, permutation in enumerate(permutation_list):
        split_spaces = []
        split_space_inside = None

        for p_count, (index, ul) in enumerate(permutation):
            if ul == 'l':
                outer1_lower = space_tobe_split_lower[index]
                outer1_lower_eqsign = space_tobe_split_lower_eqsign[index]
                outer1_upper = no_sol_lower[index]

                if no_sol_lower_eqsign[index] == 'with_equal':
                    outer1_upper_eqsign = 'without_equal'
                elif no_sol_lower_eqsign[index] == 'without_equal':
                    outer1_upper_eqsign = 'with_equal'
                else:
                    print('Error: an unexpected eqsign identifier is used',
                          file=sys.stderr)
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

                    for (i2, ul2) in permutation[:p_count]:
                        if i2 != index:
                            if ul2 == 'l':
                                new_space1_lower[i2] = no_sol_lower[i2]
                                new_space1_lower_eqsign[i2] = no_sol_lower_eqsign[i2]
                            elif ul2 == 'u':
                                new_space1_upper[i2] = no_sol_upper[i2]
                                new_space1_upper_eqsign[i2] = no_sol_upper_eqsign[i2]
                            else:
                                print("ERROR: ul has an unexpected value",
                                      file=sys.stderr)
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
                    print('Error: an unexpected eqsign identifier is used',
                          file=sys.stderr)
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

                    for (i2, ul2) in permutation[:p_count]:
                        if i2 != index:
                            if ul2 == 'l':
                                new_space2_lower[i2] = no_sol_lower[i2]
                                new_space2_lower_eqsign[i2] = no_sol_lower_eqsign[i2]
                            elif ul2 == 'u':
                                new_space2_upper[i2] = no_sol_upper[i2]
                                new_space2_upper_eqsign[i2] = no_sol_upper_eqsign[i2]
                            else:
                                print("ERROR: ul has an unexpected value",
                                      file=sys.stderr)
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
                print("ERROR: ul has an unexpected value", file=sys.stderr)
                assert False

        # finish splitting for one permutation
        score, satisfiability_list = evaluate_splitting(
            split_spaces, split_space_inside, solver, input_array, input_used_or_not)

        score_list.append(score)
        split_result_list.append(
            (split_spaces, split_space_inside, satisfiability_list))

    (best_split_spaces,
     best_split_space_inside,
     best_satisfiability_list) = split_result_list[
        score_list.index(min(score_list))]

    return best_split_spaces, best_split_space_inside, best_satisfiability_list


def split_violation_space(space, lower_limit_tmp, upper_limit_tmp,
                          solver, input_array, input_used_or_not, conf):

    lower_eqsign = ['with_equal'] * len(space[0])
    upper_eqsign = ['with_equal'] * len(space[1])

    space_tobe_split = (space[0], lower_eqsign, space[1], upper_eqsign)
    no_solution_spaces_fixed = space[2]

    split_violation_spaces = []
    no_violation_spaces = []
    loop_index = 1

    while check_splitting(
            space_tobe_split, lower_limit_tmp, upper_limit_tmp, conf.vol_ratio,
            input_used_or_not) and loop_index <= len(no_solution_spaces_fixed):

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
            solver, conf.permutation_list_max,
            input_array, input_used_or_not)

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


def calc_volume(lower_limit, upper_limit, input_used_or_not):
    vol = 1
    for index in range(len(lower_limit)):
        if input_used_or_not[index]:
            len_of_side = abs(upper_limit[index] - lower_limit[index])
            vol = vol * len_of_side

    return vol


def get_z3_solution(solver_model, var_array, var_type, fraction_cast=False,
                    default=None):

    solution_array = [default] * len(var_array)

    for index in range(len(var_array)):
        z3obj = solver_model[var_array[index]]

        if z3obj is not None:

            if 'int' in str(var_type[index]):
                value = int(z3obj.as_long())

            elif 'float' in str(var_type[index]):
                if fraction_cast:
                    value = float(z3obj.as_fraction())
                else:
                    value = z3obj.as_fraction()
            else:
                print("Error: input_type is not defined", file=sys.stderr)
                assert False

            solution_array[index] = value

    return solution_array


def print_violation(input_array, output_array,
                    input_array_converted, output_array_converted, input_type,
                    model, solver, input_name, conf):

    solution = solver.model()

    # Get z3 input
    z3_input = get_z3_solution(solution, input_array, input_type, default=0)
    if conf.conversion:
        z3_input_converted = get_z3_solution(solution, input_array_converted,
                                             input_type, default=0)

    else:
        z3_input_converted = []

    # Get z3 output
    z3_output = get_z3_solution(solution, output_array,
                                ['float' for _ in range(len(output_array))],
                                fraction_cast=True, default=0)

    if conf.conversion:
        z3_output_converted = get_z3_solution(
            solution, output_array_converted,
            ['float' for _ in range(len(output_array_converted))],
            fraction_cast=True, default=0)

    else:
        z3_output_converted = []

    # Create xgboost input
    input_data_list = get_z3_solution(solution, input_array, input_type,
                                      fraction_cast=True, default=0)

    if conf.conversion:
        input_data_list_converted = get_z3_solution(
            solution, input_array_converted, input_type,
            fraction_cast=True, default=0)

    else:
        input_data_list_converted = []

    pd.options.display.max_columns = None
    pd.options.display.width = 3500
    pd.options.display.precision = 12

    df_input_nd_converted = None
    class_probs = None
    predict_output_converted = None

    # predict
    df_input_nd = pd.DataFrame([input_data_list], columns=input_name)
    predict_output = model.predict(df_input_nd)

    if conf.conversion:
        df_input_nd_converted = pd.DataFrame(
            [input_data_list_converted], columns=input_name)
        predict_output_converted = model.predict(df_input_nd_converted)

    if conf.mode == _CLASSIFIER:
        input_dmatrix = DMatrix(df_input_nd, missing=model.missing,
                                nthread=model.n_jobs)
        ntree_limit = getattr(model, "best_ntree_limit", 0)
        class_probs = model.get_booster().predict(input_dmatrix,
                                                  output_margin=False,
                                                  ntree_limit=ntree_limit,
                                                  validate_features=True)

    # z3
    _log_writer(_LOG_MSG_DEBUG_INPUT_VALUE.format(_Z3, z3_input), log_file)
    _log_writer(_LOG_MSG_INPUT_VALUE.format(
        _Z3, '\n' + str(pd.DataFrame(
            [[float(value) if isinstance(value, Fraction) else value
              for value in z3_input]], columns=input_name))), log_file)

    if conf.conversion:
        _log_writer("", log_file)
        _log_writer(_LOG_MSG_DEBUG_INPUT_VALUE.format(
            _Z3 + "(Converted)", z3_input_converted), log_file)
        _log_writer(_LOG_MSG_INPUT_VALUE.format(
            _Z3 + "(Converted)",
            '\n' + str(pd.DataFrame(
                [[float(value) if isinstance(value, Fraction) else value
                 for value in z3_input_converted]], columns=input_name))),
            log_file)

    _log_writer("", log_file)

    if conf.mode == _REGRESSOR:
        _log_writer(_LOG_MSG_OUTPUT_VALUE.format(_Z3, z3_output), log_file)
        if conf.conversion:
            _log_writer("", log_file)
            _log_writer(_LOG_MSG_OUTPUT_VALUE.format(
                _Z3 + "(Converted)", z3_output_converted), log_file)
    else:
        _log_writer(
            _LOG_MSG_OUTPUT_VALUE.format(
                _Z3,
                "probability: {0}, class: {1}".format(
                    soft_max(z3_output), _get_category_output(z3_output))),
            log_file)

    _log_writer("", log_file)

    # xgboost
    _log_writer(_LOG_MSG_INPUT_VALUE.format(
        _XGBOOST, '\n' + str(df_input_nd)), log_file)

    if conf.conversion:
        _log_writer("", log_file)
        _log_writer(_LOG_MSG_INPUT_VALUE.format(
            _XGBOOST + "(Converted)", '\n' + str(df_input_nd_converted)),
            log_file)

    _log_writer("", log_file)

    if conf.mode == _REGRESSOR:
        _log_writer(
            _LOG_MSG_OUTPUT_VALUE.format(_XGBOOST, predict_output), log_file)

        if conf.conversion:
            _log_writer("", log_file)
            _log_writer(
                _LOG_MSG_OUTPUT_VALUE.format(
                    _XGBOOST + "(Converted)", predict_output_converted),
                log_file)

    else:
        _log_writer(_LOG_MSG_OUTPUT_VALUE.format(
            _XGBOOST,
            "probability: {0}, class: {1}".format(class_probs, predict_output)),
            log_file
        )

    _log_writer("", log_file)


def soft_max(value):
    # softmax
    exp_z3_x = np.exp([value])
    return exp_z3_x / np.sum(np.exp([value]), axis=1, keepdims=True)


def _get_category_output(z3_outputs):
    # softmax
    z3_y = soft_max(z3_outputs)

    # Return argmax
    return np.argmax(z3_y)


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
    with open(file_path, 'r', encoding="utf-8") as rs:
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

_LOG_OFF = False


def _log_writer(msg, out_stream=sys.stdout):
    if isinstance(out_stream, list):
        for out in out_stream:
            if _LOG_OFF:
                if out == sys.stdout:
                    print(msg, file=out, flush=True)
            else:
                print(msg, file=out, flush=True)
    else:
        if _LOG_OFF:
            if out_stream == sys.stdout:
                print(msg, file=out_stream, flush=True)
        else:
            print(msg, file=out_stream, flush=True)


def print_extent(lower_extent, upper_extent,
                 input_name, input_used_or_not):
    _log_writer('Violation Range', [log_file, sys.stdout])

    _log_writer('Range:', [log_file, sys.stdout])
    for index in range(len(upper_extent)):
        # if input_used_or_not[index]:
        if isinstance(lower_extent[index], Fraction) or isinstance(upper_extent[index], Fraction):
            _log_writer(
                '  {0} : {1} ({2})<= to <= {3} ({4})'.format(
                    input_name[index],
                    lower_extent[index], float(lower_extent[index]),
                    upper_extent[index], float(upper_extent[index])),
                [log_file, sys.stdout]
            )
        else:
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


def check_violation_range(solver, violation_spaces_list,
                          input_array, input_name, upper_extent, lower_extent):

    if len(violation_spaces_list) is 0:
        return True

    solver.push()
    try:
        for violation_spaces in violation_spaces_list:
            not_constraint_extent = Not(
                And(
                    And([input_array[i] <= violation_spaces[1][i]
                         for i in range(len(input_array))]),
                    And([input_array[i] >= violation_spaces[0][i]
                         for i in range(len(input_array))])
                )
            )
            solver.add(not_constraint_extent)

        constraint_extent = And(
            And([input_array[i] <= upper_extent[i]
                 for i in range(len(input_array))]),
            And([input_array[i] >= lower_extent[i]
                 for i in range(len(input_array))])
        )
        solver.add(constraint_extent)

        r = check_solver(solver)
        if r == unsat:
            _log_writer("The range of the violation is completely covered",
                        [sys.stderr, log_file])

            print_extent(lower_extent, upper_extent, input_name, None)

            return False

    finally:
        solver.pop()

    return True


def add_regressor(solver, sub_c_array, model_index, base_score, conversion=False):
    output_array = []
    c_sum = add_trees_relation_regressor(
        sub_c_array, solver, conversion=conversion)

    if not conversion:
        output_array.append(Real(_OUTPUT + str(model_index)))
    else:
        output_array.append(
            Real(_OUTPUT_CONVERTED + str(model_index)))
    solver.add(
        output_array[model_index] == (c_sum + base_score))

    return output_array


def add_conversion(input_array, output_array,
                   input_array_converted, output_array_converted,
                   solver, conv_path, conv_name):
    spec = iu.spec_from_file_location(conv_name, conv_path)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)
    exec('{0} = getattr(module, "{0}")'.format(conv_name))

    exec('convert_x = {0}(input_array)'.format(conv_name))

    print_solver = Solver()
    for i in range(len(input_array)):
        conv_exp = eval('convert_x[i] == input_array_converted[i]')
        solver.add(conv_exp)
        print_solver.add(conv_exp)

    _log_writer('===== conversion constraints =====', log_file)
    _log_writer(print_solver.sexpr(), log_file)

    del print_solver


class ConfInfo(object):

    def __init__(self, path):
        if not Path(path).exists():
            raise FileNotFoundError("{} is not found".format(path))

        self.conf_dir = Path(path).absolute().parent
        self.path = path

        # set default values
        self.search_range_ratio = 100
        self.buffer_range_ratio = None
        self.data_list_path = None
        self.prop_path = None
        self.cond_path = None
        self.system_time_out = None
        self.search_time_out = None
        self.extend_arg = _SEPARATE
        self.splitting = False
        self.vol_ratio = None
        self.permutation_list_max = None
        self.mode = _REGRESSOR
        self.conversion = False
        self.conv_file_path = None
        self.conv_func_name = None

        self.load_config()

    def load_config(self):
        with open(self.path, 'r') as rs:
            data_list = json.load(rs)

        # Check config format and get value
        if not isinstance(data_list, dict):
            raise TypeError('config type is dict')

        # Get search range ratio
        if _SEARCH_RANGE_RATIO in data_list:
            self.search_range_ratio = data_list[_SEARCH_RANGE_RATIO]

            if not (isinstance(self.search_range_ratio, int)
                    or isinstance(self.search_range_ratio, float)):
                raise TypeError(
                    '"{}" type is int or float'.format(_SEARCH_RANGE_RATIO))

        if self.search_range_ratio <= 1:
            raise ValueError('"{}" must greater 1'.format(_SEARCH_RANGE_RATIO))

        # Get buffer range ratio
        if _BUFFER_RANGE_RATIO in data_list:
            self.buffer_range_ratio = data_list[_BUFFER_RANGE_RATIO]

            if not (isinstance(data_list[_BUFFER_RANGE_RATIO], int)
                    or isinstance(data_list[_BUFFER_RANGE_RATIO], float)):
                raise TypeError(
                    '"{}" type is int or float'.format(_BUFFER_RANGE_RATIO))

        if self.buffer_range_ratio == 0:
            self.buffer_range_ratio = None

        # Get Property file path
        self.prop_path = _get_conf_path_value(
            self.conf_dir, _PROP_PATH, data_list)

        # Get Data list file path
        self.data_list_path = _get_conf_path_value(
            self.conf_dir, _DATA_LIST_PATH, data_list)

        # Get Condition file path
        if _COND_PATH in data_list:
            self.cond_path = _get_conf_path_value(
                self.conf_dir, _COND_PATH, data_list)

        # Get System Time out
        if _SYSTEM_TIMEOUT in data_list:
            self.system_time_out = data_list[_SYSTEM_TIMEOUT]

            if not (isinstance(self.system_time_out, int)
                    or isinstance(self.system_time_out, float)):
                raise TypeError(
                    '"{}" type is int or float'.format(_SYSTEM_TIMEOUT))

        # Get Search Function Time out
        if _SEARCH_TIMEOUT in data_list:
            self.search_time_out = data_list[_SEARCH_TIMEOUT]

            if not (isinstance(self.search_time_out, int)
                    or isinstance(self.search_time_out, float)):
                raise TypeError(
                    '"{}" type is int or float'.format(_SEARCH_TIMEOUT))

        # Get Extend Arg
        if _EXTEND_ARG in data_list:
            self.extend_arg = data_list[_EXTEND_ARG]

        if self.extend_arg not in [_SEPARATE, _SAMETIME]:
            raise ValueError('"{0}" value is "{1}" or "{2}"'.format(
                _EXTEND_ARG, _SEPARATE, _SAMETIME))

        # Get splitting switch
        self.splitting = data_list.get(_SPLITTING, False)

        if not isinstance(self.splitting, bool):
            raise TypeError('"{}" type is bool'.format(_SPLITTING))

        if self.splitting:
            # Get vol ratio
            if _VOL_RATIO not in data_list:
                self.vol_ratio = 100

            elif not (isinstance(data_list[_VOL_RATIO], int)
                      or isinstance(data_list[_VOL_RATIO], float)):
                raise TypeError('"{}" type is int or float'.format(_VOL_RATIO))

            else:
                self.vol_ratio = data_list[_VOL_RATIO]

            # Get permutation list_max
            if _PERMUTATION_LIST_MAX not in data_list:
                self.permutation_list_max = 10

            elif not (isinstance(data_list[_PERMUTATION_LIST_MAX], int)
                      or isinstance(data_list[_PERMUTATION_LIST_MAX], float)):
                raise TypeError(
                    '"{}" type is int or float'.format(_PERMUTATION_LIST_MAX))
            else:
                self.permutation_list_max = data_list[_PERMUTATION_LIST_MAX]

        # Get Mode (default regressor)
        if _MODE in data_list:
            self.mode = data_list[_MODE]

            if self.mode not in [_REGRESSOR, _CLASSIFIER]:
                raise ValueError('"{0}" value is "{1}" or "{2}"'.format(
                    _MODE, _REGRESSOR, _CLASSIFIER))

        # Get conversion
        self.conversion = data_list.get(_CONVERSION, False)

        if not isinstance(self.conversion, bool):
            raise TypeError('"{}" type is bool'.format(_CONVERSION))

        if self.conversion:
            if _CONV_PATH in data_list and _CONV_NAME in data_list:
                self.conv_file_path = _get_conf_path_value(
                    self.conf_dir, _CONV_PATH, data_list)
                self.conv_func_name = data_list[_CONV_NAME]

            else:
                raise ValueError(
                    'Both "{0}" and "{1}" need to be defined.'.format(
                        _CONV_PATH, _CONV_NAME
                    ))

    def __str__(self):

        str_list = []
        str_format = "{0} : {1}"
        str_list.append(str_format.format(_SEARCH_RANGE_RATIO,
                                          self.search_range_ratio))
        str_list.append(str_format.format(_BUFFER_RANGE_RATIO,
                                          self.buffer_range_ratio))
        str_list.append(str_format.format(_DATA_LIST_PATH, self.data_list_path))
        str_list.append(str_format.format(_PROP_PATH, self.prop_path))
        str_list.append(str_format.format(_COND_PATH, self.cond_path))
        str_list.append(str_format.format(_SYSTEM_TIMEOUT,
                                          self.system_time_out))
        str_list.append(str_format.format(_SEARCH_TIMEOUT,
                                          self.search_time_out))
        str_list.append(str_format.format(_EXTEND_ARG,  self.extend_arg))
        str_list.append(str_format.format(_SPLITTING, self.splitting))
        str_list.append(str_format.format(_VOL_RATIO, self.vol_ratio))
        str_list.append(str_format.format(_PERMUTATION_LIST_MAX,
                                          self.permutation_list_max))
        str_list.append(str_format.format(_MODE, self.mode))
        str_list.append(str_format.format(_CONVERSION, self.conversion))
        str_list.append(str_format.format(_CONV_PATH, self.conv_file_path))
        str_list.append(str_format.format(_CONV_NAME, self.conv_func_name))

        return '\n'.join(str_list) + '\n'


def main(model_paths, dataset, config_path):

    if model_paths is None or config_path is None:
        raise ArgumentError("")

    cur_dir = Path(__file__).absolute().parent

    start_time = time.time()

    time_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = cur_dir.joinpath("log", str(time_id + 'log'))
    log_count = 0
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
    log_file_name = str(log_dir.joinpath('log_file_{0}.txt'.format(log_count)))

    try:
        global log_file
        log_file = open(log_file_name, 'a', encoding='utf-8')

        # Load config
        conf = ConfInfo(config_path)

        _log_writer(str(conf), [sys.stdout, log_file])

        (input_name, output_name, cont_value_flag,
         input_type, uppers, lowers) = _load_data_list(conf.data_list_path)

        if conf.mode == _REGRESSOR and len(output_name) != 1:
            raise ValueError(
                "The number of models and "
                "the number of outputs are different.")

        lower_limit, upper_limit = cast_upperlower(lowers, uppers, input_type)

        for index, upper in enumerate(upper_limit):
            lower = lower_limit[index]
            if upper is None:
                print(_LOG_MSG_WARN_UPPER_LIMIT.format(input_name[index]))

            if lower is None:
                print(_LOG_MSG_WARN_LOWER_LIMIT.format(input_name[index]))

        input_array = dec_exp_vars_rev(input_type)

        if conf.conversion:
            input_array_converted = dec_exp_vars_rev(
                input_type, conversion=True)
        else:
            input_array_converted = []

        for index, name in enumerate(input_name):
            print('{0} : {1} : {2}'.format(input_array[index], name,
                                           input_type[index]))

        with open(model_paths, 'rb') as model_file:
            model = pickle.load(model_file)

        solver = Solver()

        output_array = []
        output_array_converted = []
        input_used_or_not = [False] * len(input_array)

        if False:
            model = model.best_estimator_
            booster = model.get_booster()

        else:
            booster = model.get_booster()

        # model dump(json)
        tree_tmp_dir = cur_dir.joinpath('tree_tmp')
        if not tree_tmp_dir.exists():
            tree_tmp_dir.mkdir(parents=True, exist_ok=True)

        tree_name = str(tree_tmp_dir.joinpath('tree.json'))
        booster.dump_model(tree_name, dump_format='json')

        # for num in range(
        #         len(booster.get_dump(tree_name, dump_format='json'))):
        #     graph1 = xgb.to_graphviz(model, num_trees=num)
        #     graph1.format = 'png'
        #     graph1.render('tree_' + str(num))

        # load model information(json)
        with open(tree_name, 'r') as fjson:
            djson = json.load(fjson)

        for tree in djson:
            input_name_replacer(tree, input_name)

        # Add trees consts
        sub_c_array = add_trees_consts_to_solver(djson, input_array,
                                                 input_type, solver,
                                                 input_used_or_not)
        # Get model base score
        base_score = model.get_xgb_params()['base_score']

        if conf.mode == _REGRESSOR:
            c_sum = add_trees_relation_regressor(sub_c_array, solver)

            output_array.append(Real(_OUTPUT))
            solver.add(output_array[0] == (c_sum + base_score))

            # convert
            if conf.conversion:
                sub_c_array_converted = add_trees_consts_to_solver(
                    djson, input_array_converted, input_type, solver,
                    input_used_or_not,
                    conversion=True)

                c_sum_converted = add_trees_relation_regressor(
                    sub_c_array_converted, solver, conversion=True)

                output_array_converted.append(Real(_OUTPUT_CONVERTED))
                solver.add(
                    output_array_converted[0] == (c_sum_converted + base_score)
                )

                add_conversion(
                    input_array, output_array,
                    input_array_converted, output_array_converted,
                    solver, conf.conv_file_path, conf.conv_func_name)

        else:
            # classifier
            category = len(output_name)
            output_array = [Real(_OUTPUT + str(i)) for i in range(category)]
            add_trees_relation_classifier(
                output_array, sub_c_array, category, base_score, solver)

        # Warning not used variables
        for index, or_not in enumerate(input_used_or_not):
            if or_not is False:
                _log_writer(_LOG_MSG_WARN_VARIAVLES.format(input_name[index]),
                            [sys.stdout, log_file])

        # Add property
        add_property(
            input_array, output_array,
            input_array_converted, output_array_converted,
            input_name, output_name, solver, conf.prop_path)

        # Add upper, lower limit
        add_upperlower_constraints(input_array, solver,
                                   upper_limit, lower_limit)

        if conf.cond_path:
            # Add other constraints
            add_other_constraints(input_array, output_array,
                                  input_name, output_name,
                                  solver, conf.cond_path)

        solver.push()

        violation_spaces_list = []
        violation_num = 0

        summary_extent = []

        while True:
            print("\nVerification Starts")
            satisfiability = check_solver(solver)

            if satisfiability == sat:
                violation_num += 1
                print('Violating input value exists: {}th'.format(violation_num))

                _log_writer(
                    'Range Extraction Starts', [log_file, sys.stdout])
                print_violation(input_array, output_array,
                                input_array_converted, output_array_converted,
                                input_type, model, solver, input_name, conf)

                (lower_extent, upper_extent, no_solution_spaces_fixed,
                 time_out_flag, diff_list) = find_extents_rev(
                    solver, model, lower_limit, upper_limit,
                    input_array, output_array,
                    input_array_converted, output_array_converted, input_type,
                    cont_value_flag,
                    input_name, input_used_or_not, start_time, conf
                )

                # Check violation range
                if not check_violation_range(solver, violation_spaces_list,
                                             input_array, input_name,
                                             upper_extent, lower_extent):
                    break

                print_extent(
                    lower_extent, upper_extent, input_name, input_used_or_not)

                summary_extent.append((lower_extent, upper_extent))

                if time_out_flag:
                    return

                buff_list = []

                if conf.buffer_range_ratio is not None:
                    for index in range(len(input_array)):
                        buff = calc_diff(lower_limit[index], upper_limit[index],
                                         input_type[index],
                                         conf.buffer_range_ratio)

                        buff_list.append(buff)

                    for index, buff in enumerate(buff_list):
                        lower_extent[index] -= buff
                        upper_extent[index] += buff

                violation_spaces_list.append(
                    (tuple(lower_extent),
                     tuple(upper_extent),
                     tuple(no_solution_spaces_fixed)))

                constraint_extent = And(
                    And([input_array[i] <= upper_extent[i]
                         for i in range(len(input_array))]),
                    And([input_array[i] >= lower_extent[i]
                         for i in range(len(input_array))])
                )

                solver.add(
                    Not(constraint_extent))

            else:
                print('No violating input value exists')
                break

            # 10M >= log size, Next file
            if os.path.getsize(log_file_name) >= 10 * 1024 * 1024:
                log_file.close()
                log_count = log_count + 1
                log_file_name = str(
                    log_dir.joinpath('log_file_{0}.txt'.format(log_count)))

                log_file = open(log_file_name, 'a', encoding='utf-8')

        solver.pop()

        _log_writer("\nThe number of the violations ranges is {0}\n".format(
            violation_num), [sys.stdout, log_file])

        if conf.splitting and conf.extend_arg == _SEPARATE:
            split_result_list = []

            print('\nSplitting Starts')

            if len(violation_spaces_list) > 0:

                for num, space in enumerate(violation_spaces_list):

                    print('Splitting {0}th violation range'.format(num))

                    (split_violation_spaces,
                     split_no_violation_spaces) = split_violation_space(
                        space, lower_limit, upper_limit, solver,
                        input_array, input_used_or_not, conf)

                    split_result = (
                        split_violation_spaces, split_no_violation_spaces)
                    split_result_list.append(split_result)

        solver.reset()

        elapsed_time = time.time() - start_time

        log_file.close()
        log_file_name = str(
            log_dir.joinpath('log_file_result.txt'))

        log_file = open(log_file_name, 'a', encoding='utf-8')

        _log_writer('\n===============Result===============',
                    [sys.stdout, log_file])

        # print summary extent
        for i, (le, ue) in enumerate(summary_extent):
            _log_writer("# {0}".format(i), [sys.stdout, log_file])
            print_extent(
                le, ue, input_name, input_used_or_not)
            _log_writer("", [sys.stdout, log_file])

        if conf.splitting and conf.extend_arg == _SEPARATE:
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
            [sys.stdout, log_file])
        print_extent(space_before_splitting_lower,
                     space_before_splitting_upper,
                     input_name, input_used_or_not)

        for i, elm in enumerate(split_result[0]):
            _log_writer(
                '\n===============Split Spaces %d===============' % i,
                [sys.stdout, log_file])
            print_split_spaces(elm)

        before_volume = calc_volume(space_before_splitting_lower,
                                    space_before_splitting_upper,
                                    input_used_or_not)
        _log_writer(
            '\n===============Before splitting Volume===============\n'
            '{}\n'.format(before_volume), [sys.stdout, log_file])

        split_volume_sum = 0
        for i, elm in enumerate(split_result[0]):
            split_volume = calc_volume(elm[0], elm[2], input_used_or_not)
            split_volume_sum = split_volume_sum + split_volume
            _log_writer(
                '===============After splitting Volume {0}===============\n'
                '{1}\n'.format(i, split_volume), [sys.stdout, log_file])

        _log_writer(
            '===============After splitting Volume Sum===============\n'
            '{}\n'.format(split_volume_sum), [sys.stdout, log_file])

        if split_volume_sum > 0 and before_volume > 0:
            _log_writer(
                'After splitting Volume Sum / Before splitting Volume : {}\n'.format(
                    split_volume_sum / before_volume), [sys.stdout, log_file])


if __name__ == '__main__':
    model_path = sys.argv[1]
    conf_path = sys.argv[2]

    main(model_path, None, conf_path)
