# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import sqlite3
import webbrowser

import pandas as pd

from contextlib import closing
from pathlib import Path

meta_text = \
    '<meta http-equiv="Content-Type" content="text/html; charset=utf8">\n'

sign_meaning = {
  0: "Speed Limit (20)",
  1: "Speed Limit (30)",
  2: "Speed Limit (50)",
  3: "Speed Limit (60)",
  4: "Speed Limit (70)",
  5: "Speed Limit (80)",
  6: "Limit End (80)",
  7: "Speed Limit (100)",
  8: "Speed Limit (120)",
  9: "No Overtaking",
  10: "No Overtaking by 3.5t and Larger Vehicles",
  11: "Temporary Priority",
  12: "Priority Road",
  13: "Priority Road Ahead- Slow Down",
  14: "Stop",
  15: "Vehicle Passage Prohibited",
  16: "Passage of Vehicles with Loading Capacity 3.5t or More Prohibited",
  17: "No Entry",
  18: "Danger",
  19: "Left Curve Ahead",
  20: "Right Curve Ahead",
  21: "Continuous Curves Ahead, Starting with Left Curve",
  22: "Bump Ahead",
  23: "Caution Slippery",
  24: "Right Lane Ends",
  25: "Construction Site",
  26: "Traffic Light Ahead",
  27: "Caution for Pedestrians",
  28: "Caution for Children Crossing",
  29: "Caution for Bicycles",
  30: "Caution for Freezing",
  31: "Caution for Animal Crossing",
  32: "Limit Ends",
  33: "Designated Direction Only (Right Turn Ahead)",
  34: "Designated Direction Only (Left Turn Ahead)",
  35: "Designated Direction Only (Straight Ahead)",
  36: "Designated Direction Only (Straight or Right Turn Ahead)",
  37: "Designated Direction Only (Straight or Left Turn Ahead)",
  38: "Passage Instructions (Right Side)",
  39: "Passage Instructions (Left Side)",
  40: "Roundabout",
  41: "Limit End (No Overtaking)",
  42: "Limit End (No Overtaking by3.5t and Larger Vehicles)"
}

_INDEX = 'index'
_PREDICT_LABELS = 'predict_labels'
_EVAL_VALUES = 'eval_values'
_OUT_RANGES = 'out_ranges'
_IMAGE = 'image'
_SIGN_MEANING_PREDICT = 'sign_meaning_predict'
_PATH = 'path'
_LINK = 'link'
_lib_dir = str(Path(__file__).parent.absolute())
_NO_IMAGE = 'NoImage.jpg'
_STYLE_CSS = 'style.css'
_HTML = str(Path(_lib_dir).joinpath('html'))
_IMAGES = str(Path(_HTML).joinpath('images'))


def create_dataformat(predicts, model_output_ranges):
    eval_values = []
    indexes = []
    out_ranges = []

    for pred, model_output_range in zip(predicts, model_output_ranges):
        model_output_range.is_in_range(pred)
        indexes.append(model_output_range.index)
        eval_values.append(model_output_range.eval_value)
        out_ranges.append(','.join(map(str, model_output_range.output_range)))

    data = {
        _INDEX: indexes,
        _PREDICT_LABELS: predicts,
        _EVAL_VALUES: eval_values,
        _OUT_RANGES: out_ranges
    }

    df = pd.DataFrame(data)

    return df


def to_sql(df, db_name, table_name):
    with closing(sqlite3.connect(db_name)) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def from_sql(db_name, table_name):
    query = "select * from {}".format(table_name)
    with closing(sqlite3.connect(db_name)) as conn:
        df = pd.read_sql(query, conn)
        return df


def _get_image_paths(indexes, root_path):
    d = {_PATH: []}
    if root_path.exists():

        d[_PATH] = [
            'images/%d.jpg' % index
            if root_path.joinpath('%d.jpg' % index).exists()
            else _NO_IMAGE for index in indexes
        ]
    else:
        d[_PATH] = [_NO_IMAGE] * len(indexes)

    return d


def _set_image_paths(master_df):
    indexes = master_df[_INDEX]
    root_path = Path(_IMAGES)

    d = _get_image_paths(indexes, root_path)

    df = pd.DataFrame(d)
    df[_IMAGE] = df[_PATH].map(
        lambda s: "<img src='{}' height='100' width='100'/>".format(s))

    master_df[_IMAGE] = df[_IMAGE].values


def report(master_df):
    _set_image_paths(master_df)

    master_df[_LINK] = master_df[_PREDICT_LABELS].map(
        lambda s:
        '<a href="report_label_{}.html" target="new"/>'
        'Image of test result "Success"</a>'.format(s))

    master_df = master_df.sort_values(by=[_PREDICT_LABELS, _INDEX])

    master_df = master_df.drop(_INDEX, axis=1)

    master_df = master_df.loc[:, [_IMAGE,
                                  _SIGN_MEANING_PREDICT,
                                  _OUT_RANGES,
                                  _LINK]]
    master_df = master_df.rename(columns={
        _IMAGE: 'Image of test result "Inconsistent"',
        _SIGN_MEANING_PREDICT: 'Target DNN prediction result',
        _OUT_RANGES: 'Acceptable prediction result'
    })

    # Output HTML
    old_colwidth = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)
    table = master_df.to_html(escape=False, index=False, justify='center')
    # create html
    with open(str(Path(_HTML).joinpath('report.html')), "w") as f:
        f.write(meta_text)
        f.write('<title>Result list(Failure only)</title>\n')
        f.write(table)

    pd.set_option('display.max_colwidth', old_colwidth)

    return Path(_HTML).joinpath('report.html')


def report_sub(master_df, label):
    _set_image_paths(master_df)
    image_list = master_df[_IMAGE]

    master_df = master_df.drop(_INDEX, axis=1)

    master_df = master_df.loc[:, [_IMAGE,
                                  _SIGN_MEANING_PREDICT,
                                  _OUT_RANGES]]

    master_df = master_df.rename(columns={
        _IMAGE: 'Image of test result "Success"',
        _SIGN_MEANING_PREDICT: 'Target DNN prediction result',
        _OUT_RANGES: 'Acceptable prediction result'
    })

    old_colwidth = pd.get_option('display.max_colwidth')
    pd.set_option('display.max_colwidth', -1)

    table = master_df.to_html(escape=False, index=False, justify='center')
    with open(str(Path(_HTML).joinpath(
            "report_label_%d.html" % label)), "w") as f:

        f.write(meta_text)

        f.write('<title>Success result list_%d</title>\n' % label)
        f.write('<a href="report_label_images_%d.html" target="new"/>'
                'Image list</a>\n' % label)

        f.write(table)

    pd.set_option('display.max_colwidth', old_colwidth)

    with open(str(Path(_HTML).joinpath(
            "report_label_images_%d.html" % label)), "w") as f:

        f.write(meta_text)

        f.write('<title>Image list_%d</title>\n' % label)
        f.write('<link rel="stylesheet" '
                'type="text/css" href="{}">\n'.format(_STYLE_CSS))

        for images in [image_list[i:i+5] for i in range(0, len(image_list), 5)]:
            f.write('<div class="category-box">\n')
            for image in images:
                f.write('<div class="ch">%s</div>\n' % image)

            f.write('</div>\n')


def show_html(predicts, model_output_ranges):
    master_df = create_dataformat(predicts, model_output_ranges)


    master_df[_SIGN_MEANING_PREDICT] = master_df[_PREDICT_LABELS].map(
        lambda s: '{0}: {1}'.format(s, sign_meaning[s]))

    db_name = str(Path(_lib_dir).joinpath("traffic_signs.db"))
    table_name = "result"
    to_sql(master_df, db_name, table_name)

    df = master_df.copy()

    # Get Test result "Failed"
    df = df[df[_EVAL_VALUES] == 1]
    result = report(df)

    for label in range(43):
        tmp_df = master_df.copy()

        tmp_df = tmp_df[
            (tmp_df[_EVAL_VALUES] == 0) & (tmp_df[_PREDICT_LABELS] == label)
        ]

        report_sub(tmp_df, label)

    print(str(result))

    webbrowser.open(str(result))
