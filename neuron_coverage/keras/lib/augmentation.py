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

import cv2
import keras.backend as k
import numpy as np


def _list_range(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    return list(range(start, stop, step))


def _list_float_range(start, stop=None, step=1):
    if stop is None:
        start, stop = 0, start
    return np.arange(start, stop, step)


def _is_equal_all_type(data, type_pattern):
    """
    :type data: list or tuple
    :type type_pattern: type
    :rtype: bool

    >>> _is_equal_all_type([1, 2, 3], int)
    True
    >>> _is_equal_all_type([1.0, 2.0, 3.0], float)
    True
    >>> _is_equal_all_type([1, 2.0, 3], int)
    False
    """
    if all(isinstance(d, type_pattern) for d in data):
        return True
    else:
        return False


def _is_range(*args):
    if len(args) == 1:
        args = args[0]
        if isinstance(args, tuple):
            if len(args) == 3:
                return _list_range(args[0], args[1], args[2])
            elif len(args) == 2:
                return _list_range(args[0], args[1], args[2])
            elif len(args) == 1:
                return _list_range(args[0], args[1], args[2])
            else:
                raise TypeError("arguments too long")
        elif isinstance(*args, range):
            return list(*args)
        elif isinstance(args, int):
            return _list_range(args)
        elif isinstance(args, float):
            return _list_float_range(args)
        else:
            raise TypeError("object type is {}.\n"
                            "Use tuple or int.".format(type(args)))
    else:
        raise TypeError("arguments too long.")


class ImageGenerator:
    def __init__(self,
                 contrast_range=(0, None, 1),
                 shear_range=(0, None, 1),
                 rotation_range=(0, None, 1),
                 width_shift_range=(0, None, 1),
                 height_shift_range=(0, None, 1),
                 x_scale_range=(0, None, 1),
                 y_scale_range=(0, None, 1),
                 blur_average_range=((0, None, 1),
                                     (0, None, 1)),
                 blur_gaussian_range=((0, None, 1),
                                      (0, None, 1)),
                 blur_median_range=(0, None, 1),
                 blur_bilateral_filter_range=((0, None, 1),
                                              (0, None, 1),
                                              (0, None, 1)),
                 rescale=None,
                 data_format=None):

        if data_format is None:
            data_format = k.image_data_format()

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"`'
                ' (channel after row and column) or `"channels_first"`'
                ' (channel before row and column).Received arg: ', data_format)

        self.data_format = data_format

        if data_format == "channels_first":
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == "channels_last":
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 3

        self.contrast_range = _is_range(contrast_range)
        self.shear_range = _is_range(shear_range)
        self.rotation_range = _is_range(rotation_range)
        self.width_shift_range = _is_range(width_shift_range)
        self.height_shift_range = _is_range(height_shift_range)
        self.x_scale_range = _is_range(x_scale_range)
        self.y_scale_range = _is_range(y_scale_range)
        self.blur_average_range = [_is_range(_) for _ in blur_average_range]
        self.blur_gaussian_range = [_is_range(_) for _ in blur_gaussian_range]
        self.blur_median_range = _is_range(blur_median_range)
        self.blur_bilateral_filter_range = [_is_range(_)
                                            for _ in
                                            blur_bilateral_filter_range]
        self.rescale = rescale
        self.transform_functions = []

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.

        :type x: np.ndarray
        :rtype: np.ndarray
        """
        # if self.preprocessing_function:
        #     x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale
        return x

    def transform(self):
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        if self.contrast_range:
            cr = self.contrast_range
            self.transform_functions += (partial(contrast,
                                                 alpha=c,
                                                 channel_axis=img_channel_axis)
                                         for c in cr)

        if self.shear_range:
            sr = self.shear_range
            self.transform_functions += (partial(shear,
                                                 shear_range=s,
                                                 col_axis=img_col_axis,
                                                 row_axis=img_row_axis,
                                                 channel_axis=img_channel_axis)
                                         for s in sr)
        if self.rotation_range:
            theta = self.rotation_range
            scale = 1.0
            self.transform_functions += (partial(rotate,
                                                 theta=t,
                                                 col_axis=img_col_axis,
                                                 row_axis=img_row_axis,
                                                 scale=scale,
                                                 channel_axis=img_channel_axis)
                                         for t in theta)
        if self.width_shift_range and self.height_shift_range:
            tx = self.width_shift_range
            ty = self.height_shift_range
            self.transform_functions += (partial(shift,
                                                 tx=x,
                                                 ty=y,
                                                 col_axis=img_col_axis,
                                                 row_axis=img_row_axis,
                                                 channel_axis=img_channel_axis)
                                         for x, y in zip(tx, ty))
        if self.blur_average_range:
            baw = self.blur_average_range[0]
            bah = self.blur_average_range[1]
            self.transform_functions += (partial(blur_averaging,
                                                 width=w,
                                                 height=h,
                                                 channel_axis=img_channel_axis)
                                         for w, h in zip(baw, bah))

        if self.blur_gaussian_range:
            bgw = self.blur_gaussian_range[0]
            bgh = self.blur_gaussian_range[1]
            self.transform_functions += (partial(blur_gaussian,
                                                 width=w,
                                                 height=h,
                                                 channel_axis=img_channel_axis)
                                         for w, h in zip(bgw, bgh))
        if self.blur_median_range:
            self.transform_functions += (partial(blur_median,
                                                 kernel_size=_,
                                                 channel_axis=img_channel_axis)
                                         for _ in self.blur_median_range)

        if self.blur_bilateral_filter_range:
            size = self.blur_bilateral_filter_range[0]
            sigma_color = self.blur_bilateral_filter_range[1]
            sigma_space = self.blur_bilateral_filter_range[2]
            self.transform_functions += (partial(blur_bilateral_filter,
                                                 size=s,
                                                 sigma_color=sc,
                                                 sigma_space=ss,
                                                 channel_axis=img_channel_axis)
                                         for s, sc, ss in zip(size,
                                                              sigma_color,
                                                              sigma_space))

    def flow(self, x, shuffle=False, seed=None, transpose=False):
        """Apply the processing function set by the transform method
         for each image. Iterate the applied image and return it.

        :type x: np.ndarray
        :type shuffle: bool
        :type seed: int
        :type transpose: bool
        :rtype: generator
        """
        if seed is not None:
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.transform_functions)

        if transpose:
            for func in self.transform_functions:
                for inp in x:
                    inp = self.standardize(inp)
                    yield func(inp)
        else:
            for inp in x:
                for func in self.transform_functions:
                    inp = self.standardize(inp)
                    yield func(inp)


def brightness(x, beta, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    beta = np.array([beta])
    channel_images = [cv2.add(x_channel, beta)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def contrast(x, alpha, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    alpha = np.array([alpha])
    channel_images = [cv2.multiply(x_channel, alpha)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def shift(x, tx, ty, col_axis, row_axis, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    m = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    channel_images = [cv2.warpAffine(x_channel,
                                     m,
                                     (x.shape[col_axis], x.shape[row_axis]))
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def shear(x, shear_range, col_axis, row_axis, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])

    pt1 = 50 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 200 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 50], [pt2, pt1], [50, pt2]])

    m = cv2.getAffineTransform(pts1, pts2)

    channel_images = [cv2.warpAffine(x_channel,
                                     m,
                                     (x.shape[col_axis], x.shape[row_axis]))
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def zoom(x, sx, sy, col_axis, row_axis, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    m = np.float32([[sx, 0, 0],
                    [0, sy, 0]])
    channel_images = [cv2.warpAffine(x_channel,
                                     m,
                                     (x.shape[col_axis], x.shape[row_axis]))
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def rotate(x, theta, col_axis, row_axis, scale=1, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    ox, oy = int(x.shape[col_axis] / 2), int(x.shape[row_axis] / 2)
    r = cv2.getRotationMatrix2D((ox, oy), theta, scale)
    channel_images = [cv2.warpAffine(x_channel,
                                     r,
                                     (x.shape[col_axis], x.shape[row_axis]),
                                     flags=cv2.INTER_CUBIC)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def blur_averaging(x, width, height, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [cv2.blur(x_channel, ksize=(width, height))
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def blur_gaussian(x, width, height, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [cv2.GaussianBlur(x_channel,
                                       ksize=(width, height),
                                       sigmaX=1.2)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def blur_median(x, kernel_size, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [cv2.medianBlur(x_channel, kernel_size)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def blur_bilateral_filter(x, size, sigma_color, sigma_space, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    channel_images = [cv2.bilateralFilter(x_channel,
                                          size,
                                          sigma_color,
                                          sigma_space)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x
