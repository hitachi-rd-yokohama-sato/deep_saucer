# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

plt.style.use("ggplot")


def get_zipped_data(input_shapes, inputs, model, coverage):
    result = []
    for i in inputs:
        model.update(i)
        c1 = model.get_ignition_counter
        c2 = model.get_not_ignition_counter
        input_shape = model.input_shape
        units = model.all_units
        coverage.update(c1, c2, input_shape, units)
        result.append(coverage.get_activation_rate)
        # result.append(coverage.get_non_activation_rate)
    zipped_shapes = zip(input_shapes, result)
    return zipped_shapes


def convert_dataset_to_histogram(label_dataset, num_classes):
    """
    :type label_dataset: np.ndarray
    :type num_classes: int

    e.g.
    categorical dataset
    [[ 1.  0.]  is 1
     [ 0.  1.]  is 0
     [ 0.  1.]  is 0
     [ 1.  0.]] is 1
     Counting per category

    >>> import numpy as np
    >>> from pprint import pprint
    >>> classes = 2
    >>> num_of_dataset = 4
    >>> labels = np.array([[ 1, 0], [ 0, 1], [ 0, 1], [ 1, 0]])
    >>> counter = convert_dataset_to_histogram(labels, classes)
    >>> pprint(counter)
    Counter({0: 2, 1: 2})
    """
    labels = to_categorical(range(num_classes), num_classes)
    counter = Counter({i: label_dataset[label_dataset == label].sum()
                       for i, label in enumerate(labels)})

    return counter


def plot_histogram(counter):
    plt.bar(list(counter.keys()), list(counter.values()), align="center")
    plt.title("label vs frequency")
    plt.xlabel("label")
    plt.ylabel("frequency")
    plt.xticks()
    plt.show()


def plot_instance_counts(dataset, num_classes, name="dataset"):
    counts = convert_dataset_to_histogram(dataset, num_classes=num_classes)
    labels, values = zip(*counts.items())
    indexes = np.arange(len(labels))
    width = 0.5
    with plt.style.context(('seaborn-muted')):
        figure = plt.figure(figsize=(15, 3))
        plt.bar(indexes, values, width)
        plt.xticks(indexes + width * 0.5, labels)
        plt.xlabel('Class Label')
        plt.title('{} : Number of instance per class'.format(name))

    plt.show()


def plot_history(history):
    """ Only use during learning to the end!!

    :type history: keras.model.history
    """
    # accuracy plot
    plt.plot(history.history['acc'], "o-", label="accuracy")
    plt.plot(history.history['val_acc'], "o-", label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")
    plt.show()

    # loss plot
    plt.plot(history.history['loss'], "o-", label="loss", )
    plt.plot(history.history['val_loss'], "o-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


def random_index(data_shape, input_shape):
    return np.random.choice(np.arange(data_shape), input_shape, replace=False)
