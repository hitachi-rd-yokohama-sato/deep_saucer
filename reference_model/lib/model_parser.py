# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import re
import os
from functools import total_ordering
from itertools import product


@total_ordering
class State:
    def __init__(self, name):
        self.name = name
        count = 0
        for ix, n in enumerate(self.name):
            if n.isdecimal():
                count = ix
                break
        self.identifier = self.name[:count]
        self.index = int(self.name[count:])

    def __eq__(self, other):
        if not isinstance(other, State):
            if isinstance(other, str):
                return str(self) == other
            return NotImplemented
        return self.index == other.index and self.name == other.name

    def __lt__(self, other):
        return self.index < other.index

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.identifier) + hash(self.index)


class ReferenceModelParser:
    def __init__(self, path):
        self.path = path
        self.begin_state = None
        self.states = set()
        """
        reference_model_regex
        This is Conditional Transition Pattern
        ex:
        S0 = (i1,i2) i1 == 2 or i2 == 2 (11, 12) -> S5

        1. S0 -> Conditional transition destination
        2. (i1,i2) -> input
        3. i1 == 2 or i2 == 2 -> Transition condition
        4. (11, 12) -> Allowable output value range
        5. S5 -> State name
        """
        self.cond_trans_pattern = re.compile(
            r"^(.+)\s+=\s+\((.+)\)\s+(.*)\s+\((.*)\)\s+->\s+(.+)")
        self.conditional_transitions = []
        """
        reference_model_regex
        This is UnConditional Transition Pattern
        ex:
        S5 = -> S0

        1. S5 -> Unconditional transition destination
        2. S0 -> State name
        """
        # self.uncond_trans_pattern = re.compile(r"^(.*) = -> (.*)")
        self.uncond_trans_pattern = re.compile(r"^(.+)\s+=\s+->\s+(.+)")
        self.unconditional_transitions = []

    def parse(self):
        if os.stat(self.path).st_size == 0:
            raise IOError("File is empty.")
        with open(self.path, "r") as f:
            for sentence in f:
                cond_trans = self.cond_trans_pattern.search(sentence)
                uncond_trans = self.uncond_trans_pattern.search(sentence)
                if sentence == "":
                    break
                elif cond_trans:
                    transition_dest, params, condition, \
                        output_range, state_name = cond_trans.groups()
                    tr = Conditional(source=transition_dest, target=state_name,
                                     cond_expr=condition, labels=output_range,
                                     params=params)
                    self.states.add(tr.target)
                    self.conditional_transitions.append(tr)
                elif uncond_trans:
                    transition_dest, state_name = uncond_trans.groups()
                    tr = UnConditional(source=transition_dest,
                                       target=state_name)
                    self.states.add(tr.target)
                    self.unconditional_transitions.append(tr)
                else:
                    raise re.error("{} didn't match.".format(sentence))
        self.begin_state = sorted(self.states)[0]

    def is_empty(self):
        if not self.begin_state:
            return True
        else:
            return False

    def get_conditional_transitions(self, begin_state):
        for transition in self.conditional_transitions:
            if begin_state == transition.source:
                yield transition

    def get_model_output_ranges(self, arg_generator):
        begin_states = [self.begin_state]
        unconditional_transitions = self.unconditional_transitions

        model_output_ranges = []
        nest_model_output_ranges = []

        for idx, args in enumerate(arg_generator):
            begin_state = begin_states.pop()
            next_states = []
            nest_output_ranges = []
            for transition in self.get_conditional_transitions(begin_state):
                if transition.condition(**args):
                    mor = ModelOutputRange(index=idx,
                                           source_state=transition.source,
                                           target_state=transition.target,
                                           output_range=transition.labels)
                    model_output_ranges.append(mor)
                    next_state = mor.get_next_state(transition.target,
                                                    unconditional_transitions)
                    next_states.append(next_state)
                    nest_output_ranges.append(mor)
            begin_states = next_states
            nest_model_output_ranges.append(nest_output_ranges)
            if not begin_states:
                break

        return model_output_ranges, nest_model_output_ranges


class Transition:
    def __init__(self, source, target):
        """

        :type source: str
        :type target: str
        """
        self.source = State(self.is_blank(source))
        self.target = State(self.is_blank(target))

    @staticmethod
    def is_blank(text):
        """
        :type text: str
        :return str
        """
        if not text:
            raise ValueError("{} is blank.".format(text))
        else:
            return text


class Conditional(Transition):
    def __init__(self, source, target,
                 params, cond_expr, labels):
        """

        :type source: str
        :type target: str
        :type params: str
        :type cond_expr: str
        :type labels: str
        """
        self.params = self.split_params(params)
        self.cond_expr = cond_expr
        self.labels = self.str2set(labels)
        super(Conditional, self).__init__(source, target)

    @staticmethod
    def split_params(params):
        """

        :type params: str
        :return list[str]
        """
        params = params.split(",")
        if len(params) >= 2:
            return [p.strip() for p in params]
        else:
            return params

    @staticmethod
    def str2set(output_range):
        """
        :type output_range: str
        :return set
        """
        if output_range == "":
            output = {}
        else:
            output = set(map(int, output_range.split(",")))
        return output

    def condition(self, **kwargs):
        expression = self.cond_expr

        if kwargs.keys() < set(self.params):
            raise NameError("{} is not defined.".format(self.params))

        for arg, value in kwargs.items():
            if arg in self.params:
                expression = expression.replace(arg, str(value))
        return eval(expression)


class UnConditional(Transition):
    def __init__(self, source, target):
        super(UnConditional, self).__init__(source, target)


class ModelOutputRange:
    def __init__(self, index, source_state, target_state, output_range):
        self.index = index
        self.source_state = source_state
        self.target_state = target_state
        self.next_state = None
        self.output_range = output_range
        self.eval_value = True
        self.software_output = None
        self.weight = None

    def __str__(self):
        if self.eval_value is True and self.software_output is None:
            return "({}, {}, {}, {})".format(self.source_state,
                                             self.target_state,
                                             self.output_range,
                                             self.next_state)
        else:
            return "({}, {}, {}, {}, {}, {})".format(self.source_state,
                                                     self.target_state,
                                                     self.next_state,
                                                     self.output_range,
                                                     int(self.eval_value),
                                                     self.software_output)

    def get_next_state(self, next_state, transitions):
        """

        :type next_state: State
        :type transitions: list[UnConditional]
        """
        for t in transitions:
            if next_state == t.source:
                state = t.target
                return self.get_next_state(state, transitions)
            else:
                self.next_state = next_state
        return next_state

    def is_in_range(self, output):
        self.software_output = output
        if output in self.output_range:
            self.eval_value = False

    def is_match(self, weight):
        st = weight.state_to
        sf = weight.state_from
        if self.target_state == st and self.source_state == sf:
            return True
        else:
            return False

    def is_weight(self):
        if self.weight:
            return True
        else:
            return False

    # def __eq__(self, other):
    #     if not isinstance(other, ModelOutputRange):
    #         return NotImplementedError
    #     return all([self.index == other.index,
    #                 self.source_state == other.source_state,
    #                 self.target_state == other.target_state])
    #
    # def __hash__(self):
    #     return sum([hash(self.index),
    #                 hash(self.source_state),
    #                 hash(self.target_state)])
