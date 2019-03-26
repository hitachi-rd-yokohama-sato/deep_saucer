# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from z3 import *


class VerifyZ3:

    def __init__(self, network_struct, logic='QF_LRA'):
        self.network_struct = network_struct

        self.parser = _generate_parser()

        self.z3_obj = SolverFor(logic)

    def parse_property(self, property_path):
        # Sat when not satisfying property
        in_expr_list = _get_expr(self.network_struct.input_placeholders_info)
        for expr in in_expr_list:
            exec(expr)

        out_expr_list = _get_expr(self.network_struct.output_nodes_info)
        for expr in out_expr_list:
            exec(expr)

        express_list =[]
        with open(property_path, 'r') as f:

            for line in f:
                express = _make_express(
                    self.parser.parseString(line).asList()[0])
                express_list.append(express)

        self.z3_obj.add(eval('Not(And({}))'.format(','.join(express_list))))

    def parse_condition(self, condition_path):
        in_expr_list = _get_expr(self.network_struct.input_placeholders_info)
        for expr in in_expr_list:
            exec(expr)

        out_expr_list = _get_expr(self.network_struct.output_nodes_info)
        for expr in out_expr_list:
            exec(expr)

        with open(condition_path, 'r') as f:

            for line in f:
                condition = _make_express(
                    self.parser.parseString(line).asList()[0])
                self.z3_obj.add(eval(condition))


def _get_expr(io_inf):
    expr_list = []
    for ii in io_inf:

        in_type = ii.dtype

        for var_name in ii.var_names:
            tmp_expr_right = get_z3obj_type(in_type, var_name)
            tmp_expr = '{0}={1}("{0}")'.format(var_name,
                                               str(tmp_expr_right.sort()))
            expr_list.append(tmp_expr)

    return expr_list


def _generate_parser():
    import pyparsing as pp

    integer = pp.Word(pp.nums)
    real = pp.Combine(pp.Word(pp.nums) + "." + pp.Word(pp.nums))
    exponent = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")

    variable = pp.Word(pp.alphanums + "_")
    operand = exponent | real | integer | variable | pp.quotedString | pp.dblQuotedString

    sign_op = pp.oneOf('+ -')
    mult_op = pp.oneOf('* / %')
    plus_op = pp.oneOf('+ -')
    exp_op = pp.Literal('**')

    arith_expr = pp.infixNotation(operand,
                                  [
                                      (sign_op, 1, pp.opAssoc.RIGHT,),
                                      (mult_op, 2, pp.opAssoc.LEFT,),
                                      (plus_op, 2, pp.opAssoc.LEFT,),
                                      (exp_op, 2, pp.opAssoc.LEFT,),
                                  ])

    comparison_op = pp.oneOf("< <= > >= != ==")
    cond_expr = pp.infixNotation(arith_expr,
                                 [
                                     (comparison_op, 2, pp.opAssoc.LEFT,),
                                 ])
    not_op = pp.oneOf('!')
    and_op = pp.oneOf('&&')
    or_op = pp.oneOf('||')
    arrow_op = pp.oneOf('=>')
    complex_expr = pp.infixNotation(cond_expr,
                                    [
                                        (not_op, 1, pp.opAssoc.RIGHT,),
                                        (and_op, 2, pp.opAssoc.LEFT,),
                                        (or_op, 2, pp.opAssoc.LEFT,),
                                        (arrow_op, 2, pp.opAssoc.RIGHT,),
                                    ])
    return complex_expr


def get_z3obj_type(d_type, name):
    from z3 import Bool, Int, Real

    if d_type.is_bool:
        return Bool(name)
    elif d_type.is_integer:
        return Int(name)
    else:
        return Real(name)


def _make_express(parse_list):
    ret_str = ""

    if len(parse_list) == 1 and isinstance(parse_list[0], str):
            ret_str = parse_list[0]

    else:

        flg = False

        for parse in parse_list:

            if isinstance(parse, list):
                sub_str = _make_express(parse)
            else:
                # Del quote
                sub_str = parse.strip('"\'')

            if flg:
                sub_str = sub_str + ')'
                flg = False

            if sub_str == '!':
                ret_str = ret_str + 'Not('
                flg = True

            elif sub_str == '||':
                ret_str = 'Or(' + ret_str + ','
                flg = True

            elif sub_str == '&&':
                ret_str = 'And(' + ret_str + ','
                flg = True

            elif sub_str == '=>':
                ret_str = 'Implies(' + ret_str + ','
                flg = True

            else:
                ret_str = ret_str + sub_str

    return ret_str
