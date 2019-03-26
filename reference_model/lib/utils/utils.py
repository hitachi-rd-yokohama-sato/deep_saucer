# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as k
from keras.utils import to_categorical


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


def get_model_predicts(model, inputs):
    """
    :param model: keras.models.Sequential
    :param inputs: np.ndarray
    :return: np.ndarray
    >>> import numpy as np
    >>> from keras.models import load_model
    >>> from keras.utils import HDF5Matrix
    >>> test_path = "input_data/test_dataset_3_32_32.h5"
    >>> test_x = HDF5Matrix(test_path, "images")[:10]
    >>> model = load_model("model/vggnet_2.hdf5", compile=False)
    >>> get_model_predicts(model, test_x)
    array([16,  1, 38, 33, 11, 38, 18, 12, 25, 35], dtype=int64)
    """
    return np.argmax(model.predict(inputs), axis=1)


def plot_tiles(images, rows=5, columns=5):
    """
    Example:
    rows=5
    columns=5
    images.shape -> (25, 3, 32, 32)
    iterate (3, 32, 32)...5 * 5 plot image tiles

    :param np.array images:
    :param int rows:
    :param int columns:
    """
    pos = 1
    plt.figure(figsize=(12, 8))
    for i in range(rows * columns):
        plt.subplot(rows, columns, pos)
        img = images[i].transpose((1, 2, 0))
        plt.imshow(img)
        plt.axis("off")
        pos += 1
    plt.show()


def read_csv(path):
    """ Return list of strings per comma.

    :param str path: csv located path
    :rtype list[str]
    """
    args = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            for r in row:
                args.append(r)
        return args


def get_color_labels(labels):
    """ convert one_hot labels from 43 classes to 3 classes
    3 color classes is ...
    0 -> red
    1 -> blue
    2 -> red and blue else

    :param np.array labels: 43 classes labels
    :rtype np.array

    >>> import numpy as np
    >>> from keras.utils import to_categorical
    >>> origin_labels = to_categorical([0, 6, 40], 43)
    >>> new_labels = get_color_labels(origin_labels)
    >>> new_labels
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.]])
    """
    # labels 43 to 3(color)
    color_classes = 3
    # 0 is red, 1 is blue, 2 is else
    color_labels = [0, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0,
                    2, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 2, 1, 1, 1,
                    1, 1, 1, 1, 1, 2,
                    2]

    return convert_specific_label(color_labels,
                                  labels,
                                  color_classes)


def get_shape_labels(labels):
    """ convert one_hot labels from 43 classes to 3 classes
    3 shape classes is ...
    0 -> circle
    1 -> triangle
    2 -> circle and triangle else

    :param np.array labels: 43 classes labels
    :rtype: np.array

    >>> import numpy as np
    >>> from keras.utils import to_categorical
    >>> origin_labels = to_categorical([0, 11, 12], 43)
    >>> new_labels = get_shape_labels(origin_labels)
    >>> new_labels
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    """
    # labels 43 to 3(shape)
    shape_classes = 3
    # 0 is circle, 1 is triangle, 2 is else
    shape_labels = [0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 1,
                    2, 1, 2, 0, 0, 0,
                    1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1,
                    1, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    0]

    return convert_specific_label(shape_labels,
                                  labels,
                                  shape_classes)


def convert_specific_label(new_labels, old_labels, num_classes):
    """ get_color_labels and get_shape_labels utility function

    :param np.array new_labels:
    :param np.array old_labels:
    :param int num_classes: new_labels's num_classes
    :rtype: np.array
    """
    cat = to_categorical(new_labels, num_classes=num_classes)
    return cat[np.argmax(old_labels, axis=1)]
