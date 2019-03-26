# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import numpy as np
from itertools import repeat
from keras import backend as k
from keras.models import load_model
from keras.preprocessing.image import (apply_transform,
                                       transform_matrix_offset_center)

ColorClassificationModelPath = r"C:\cygwin64\home\spc000\python" \
                               r"\new_project\model\color_classification.h5"
ShapeClassificationModelPath = r"C:\cygwin64\home\spc000\python" \
                               r"\new_project\model\shape_classification.h5"

TrafficSignClassificationModelPath = r"C:\cygwin64\home\spc000\python" \
                                     r"\new_project\model\vggnet_2.hdf5"


class Converter:
    def __init__(self, inputs, *args):
        self.inputs = inputs
        self.funcs = args

    def flow(self):
        for func in self.funcs:
            yield func(self.inputs)


def get_channel_indexes(channel_axis=0,
                        row_axis=1,
                        col_axis=2):
    data_format = k.image_data_format()
    if data_format not in {'channels_first',
                           'channels_last'}:
        raise ValueError("Invalid data_format:", data_format)

    if data_format == "channels_first":
        return channel_axis, row_axis, col_axis
    else:
        channel_axis = 2
        row_axis = 0
        col_axis = 1
        return channel_axis, row_axis, col_axis


class ArgumentGenerator:
    def __init__(self, args, predicts):
        self.args = args
        self.predicts = predicts

    def __iter__(self):
        """

        >>> args = ["i1", "i2"]
        >>> predicts = [[0, 1, 2], [3, 4, 5]]
        >>> arg_gen = ArgumentGenerator(args, predicts)
        >>> print([sorted(_.items(), key=lambda x: x[0]) for _ in arg_gen])
        [[('i1', 0), ('i2', 3)], [('i1', 1), ('i2', 4)], [('i1', 2), ('i2', 5)]]
        """
        repeat_args = repeat(self.args, len(self.predicts[0]))
        for r, p in zip(repeat_args, zip(*self.predicts)):
            yield dict(zip(r, p))


def blur():
    pass


def rotation(x, rotation_range=90, fill_mode="nearest", cval=0.):
    channel_axis, row_axis, col_axis = get_channel_indexes()
    theta = np.deg2rad(rotation_range)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    height, width = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix,
                                                      height,
                                                      width)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def shear(x, shear_range=30, fill_mode="nearest", cval=0.):
    channel_axis, row_axis, col_axis = get_channel_indexes()
    sh = np.deg2rad(shear_range)
    shear_matrix = np.array([[1, -np.sin(sh), 0], [0, np.cos(sh), 0],
                             [0, 0, 1]])
    height, width = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix,
                                                      height,
                                                      width)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def channel_shift(x, intensities):
    channel_axis, _, _ = get_channel_indexes()
    # x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)

    channel_images = []
    for x_channel, shift in zip(x, intensities):
        apply_channel = np.clip(x_channel + shift, min_x, max_x)
        channel_images.append(apply_channel)
    x = np.stack(channel_images, axis=0)
    # x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def get_predict_result(inputs, model):
    # check function
    return np.argmax(model.predict(inputs), axis=1)


def get_color(inputs):
    model = load_model(ColorClassificationModelPath)
    return get_predict_result(inputs, model)


def get_shape(inputs):
    model = load_model(ShapeClassificationModelPath)
    return get_predict_result(inputs, model)


def get_shear_images_predict(inputs, shear_range=10):
    model = load_model(TrafficSignClassificationModelPath)
    transform_images = np.array([shear(img, shear_range) for img in inputs])
    return get_predict_result(transform_images, model)


def get_rotation_images_predict(inputs, rotation_range=20):
    model = load_model(TrafficSignClassificationModelPath)
    transform_images = np.array([shear(img, rotation_range) for img in inputs])
    return get_predict_result(transform_images, model)


def get_channel_shift_predict_images(inputs, intensities):
    model = load_model(TrafficSignClassificationModelPath)
    transform_images = np.array([channel_shift(img, intensities)
                                 for img in inputs])
    return get_predict_result(transform_images, model)
