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
import networkx as nx
from itertools import product


class Consistency:
    def __init__(self, path):
        self.path = path
        self.pattern = re.compile(r"(.+)\s+->\s+(.+)\s+/\s+(.+)")
        self.weight_models = []

    def parse(self):
        if os.stat(self.path).st_size == 0:
            raise IOError("File is empty.")
        with open(self.path, "r") as f:
            for sentence in f:
                if sentence == "":
                    break
                match = self.pattern.search(sentence)
                if match:
                    w = WeightModel()
                    w.state_from, w.state_to, w.weight = match.groups()
                    w.weight = float(w.weight)
                    self.weight_models.append(w)
                else:
                    raise re.error("{} is not matched.".format(sentence))


def get_directed_graph(last_idx, nest_output_ranges):
    graph = nx.DiGraph()
    first_idx = 0
    heads = []
    tails = []
    sources = None

    for output_ranges in nest_output_ranges:
        targets = []
        for output_range in output_ranges:
            if output_range.index == first_idx:
                heads.append(output_range)
            elif output_range.index == last_idx:
                tails.append(output_range)
            graph.add_node(output_range)
            targets.append(output_range)

        if sources:
            for src, tar in product(sources, targets):
                if src.next_state == tar.source_state:
                    graph.add_edge(src, tar)
        sources = targets
    return graph, heads, tails


def calc_error_weight(paths):
    result = 0
    for path in paths:
        result += path.eval_value * path.weight
    return result


def set_weight(paths, weights):
    for path, weight in product(paths, weights):
        if path.is_match(weight):
            path.weight = weight.weight
    return paths


def calc(graph, heads, tails, consistency):
    result = 0
    # inputs length is 1
    if len(heads) == 1 and len(tails) == 0:
        paths = set_weight(heads, consistency.weight_models)
        result += calc_error_weight(paths)
        return result
    for head, tail in product(heads, tails):
        for paths in nx.all_simple_paths(graph, source=head,
                                         target=tail):
            paths = set_weight(paths, consistency.weight_models)
            result += calc_error_weight(paths)
    return result


class WeightModel:
    def __init__(self):
        self.state_to = ""
        self.state_from = ""
        self.weight = 0

    def __str__(self):
        return "{} -> {} / {}".format(self.state_from,
                                      self.state_to,
                                      self.weight)
