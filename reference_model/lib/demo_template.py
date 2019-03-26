# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from functools import partial
from pathlib import Path

from lib.html_result import show_html
from lib.match import Consistency, get_directed_graph, calc
from lib.model_parser import ReferenceModelParser
from lib.utils.utils import plot_tiles, get_model_predicts


def debug_print(*args, debug=True):
    if debug:
        print(*args)


def is_path(path):
    """
    :param path: str
    :return: str
    """
    p = Path(path)
    if p.exists():
        return str(p)
    else:
        raise FileExistsError("{} don't exist.".format(path))


class DemoTemplate:
    def __init__(self, model, dataset,
                 reference_model, evaluation_criteria, arguments,
                 plot=True, debug=True):
        self.model = model
        self.dataset = dataset
        self.reference_model = is_path(reference_model)
        self.evaluation_criteria = is_path(evaluation_criteria)
        self.arguments = arguments
        self.plot = plot
        self.debug = debug

    def play(self):
        rows = 1
        if self.plot:
            # Show images
            plot_tiles(self.dataset, rows=rows, columns=len(self.dataset))

        log = partial(debug_print, debug=self.debug)

        log('Run Targe DNN')
        predicts = get_model_predicts(self.model, self.dataset)

        log("---------------------------------------------")
        log("DNN Predicts(ID labels)")
        for idx, p in enumerate(predicts):
            log(idx, p)

        log('Get Model Output Range (%s)' % self.reference_model)
        parser = ReferenceModelParser(self.reference_model)
        parser.parse()
        model_output_ranges, nest_output_ranges = \
            parser.get_model_output_ranges(self.arguments)

        log("---------------------------------------------")
        log("Data Converter Output(ID color and shape)")
        for idx, predict in enumerate(self.arguments):
            log(idx, sorted(predict.items()))

        log('Evaluate DNN Result with Model Output Range')
        log("---------------------------------------------")
        log("Model Output Ranges")
        for idx, pred_and_model_output_range in enumerate(
                zip(predicts, model_output_ranges)):
            pred, model_output_range = pred_and_model_output_range
            model_output_range.is_in_range(pred)
            log(idx, model_output_range.output_range)

        log("---------------------------------------------")
        log("Difference information")
        for idx, mor in enumerate(model_output_ranges):
            log(idx, mor)

        c = Consistency(self.evaluation_criteria)
        c.parse()

        graph, heads, tails = get_directed_graph(len(predicts) - 1,
                                                 nest_output_ranges)

        result = calc(graph, heads, tails, c)
        print("---------------------------------------------")
        print("The number of test cases:%d" % len(self.dataset))
        print("Inconsistent test cases:%d" % int(result))
        print("Success rate:{:.2f}".format(1 - (result / len(self.dataset))))
        print("---------------------------------------------")

        show_html(predicts, model_output_ranges)

