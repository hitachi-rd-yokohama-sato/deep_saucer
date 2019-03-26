# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
"""
GTSRB image_set is separated 43 classes
"""
import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, exposure, transform, io

Training_PATH = 'GTSRB_data/Final_Training/Images/'
Test_PATH = 'GTSRB_data/Final_Test/Images/'
Test_CSV_Path = 'GTSRB_data/GT-final_test.csv'
t_path = pathlib.Path(Training_PATH)
test_path = pathlib.Path(Test_PATH)
final_test = pathlib.Path(Test_CSV_Path)


class Image:
    def __init__(self, image_path):
        """
        :type image_path: str
        """
        self.img = io.imread(image_path)

    def show(self):
        io.imshow(self.img)
        plt.show()


def cnv2rgb(img, centre, size):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    min_side = min(img.shape[:-1])
    img = img[
          centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]
    # rescale to standard size
    img = transform.resize(img, (size, size))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def cnv2gs(img, centre, size):
    img = color.rgb2gray(img)
    min_side = min(img.shape)
    img = img[
          centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2]
    # rescale to standard size
    img = transform.resize(img, (size, size))

    return img


def rgb2flatten(img):
    flatten = img.flatten()
    return flatten


def run_image(img, n_color='rgb', size=48):
    # Histogram normalization in v channel only
    # rgb add Lightness
    """
    :type size: int
    :type n_color: str
    :type img: np.ndarray
    :rtype: np.ndarray
    """
    # central square crop
    # get minimum shape height or width
    centre = img.shape[0] // 2, img.shape[1] // 2

    if n_color == 'rgb':
        img = cnv2rgb(img, centre, size)
    elif n_color == 'gs':
        img = cnv2gs(img, centre, size)
    else:
        raise TypeError("n_color is 'rgb' or 'gs'.")

    return img


def classification(img_path):
    """
    :type img_path: pathlib.Path
    :rtype: int
    >>> classification(pathlib.Path("\path\00022\image.jpg"))
    22
    """
    return int(img_path.parts[-2])


def convert_images(classes=43, _color='rgb', size=32):
    images = []
    labels = []

    all_img_paths = get_training_image_paths(shuffle=True)

    for img_path in all_img_paths:
        img = run_image(io.imread(str(img_path)), n_color=_color, size=size)
        label = classification(img_path)
        images.append(img)
        labels.append(label)

    x = np.array(images, dtype="float32")
    y = np.eye(classes, dtype="uint8")[labels]

    return x, y


def get_test_labels(test_csv_path):
    """
    :type test_csv_path: str
    :rtype: list[int]

    The Answer(=ClassId) of test_dataset.
    get only ClassId from csv.
    e.g.
        Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId
        00000.ppm;53;54;6;5;48;49;16
                                  ^^
        00002.ppm;48;52;6;6;43;47;38
                                  ^^
        ...
    """
    labels = []
    with open(test_csv_path, "r") as f:
        reader = f.readlines()
        for row in reader[1:]:
            labels.append(int(row.split(";")[-1].strip()))

    return labels


def get_image_paths(root_path_dir, shuffle=False):
    """
    :type root_path_dir: pathlib.Path
    :type shuffle: bool
    :rtype: list[pathlib.Path]

    ex.
    root - dir -- image1.jpg
               -- image2.jpg
               -- ...
    root_path_dir <-- root/dir
    """
    all_img_paths = root_path_dir.glob("*.ppm")
    all_img_paths = list(all_img_paths)

    if shuffle:
        np.random.shuffle(all_img_paths)

    return all_img_paths


def get_training_image_paths(shuffle=True):
    """
    :type shuffle: bool
    :rtype: list[pathlib.Path]
    """
    all_img_paths = t_path.glob("*/*.ppm")
    all_img_paths = list(all_img_paths)

    if shuffle:
        np.random.shuffle(all_img_paths)

    return all_img_paths


def save2hdf5(x, y, filename):
    """
    :type x: np.array
    :param x
    e.g. rgb image(numpy array) shape 32 * 32 * 3
    :type y: np.array
    :param y
    class label
    43 classes -> 0 ~ 42
    :type filename: str
    :param filename
    e.g. "train.h5", "test.h5" ...

    """
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("images", data=x)
        hf.create_dataset("labels", data=y)


def convert2hdf5_test_dataset(classes=43):
    test = pd.read_csv(str(final_test), sep=';')
    x_test = []
    y_test = []
    for file_name, class_id in zip(list(test['Filename']),
                                   list(test['ClassId'])):
        image_path = test_path.joinpath(file_name)
        x_test.append(run_image(io.imread(str(image_path))))
        y_test.append(int(class_id))

    x = np.array(x_test, dtype="float32")
    y = np.eye(classes, dtype="uint8")[y_test]

    save2hdf5(x, y, "test.h5")


def load_dataset(filename):
    """
    :type filename: str
    """
    try:
        with h5py.File(filename, "r") as hf:
            x, y = hf['images'][:], hf['labels'][:]
        print("Loaded images from {0}".format(filename))
        print("shape: {0}".format(x.shape))
        print("dtype: {0}".format(x.dtype))
        return x, y
    except (IOError, OSError, KeyError):
        print("Couldn't load dataset from hdf5 file")


def get_random_image_path():
    all_img_paths = get_training_image_paths(shuffle=False)
    rand_index = np.random.randint(len(all_img_paths) - 1)
    return all_img_paths[rand_index]


def save_dataset(file_name, classes=43, _color='rgb', size=32):
    """
    :type size: int
    :type _color: str
    :type classes: int
    :type file_name: str
    """
    x, y = convert_images(classes, _color, size)
    save2hdf5(x, y, file_name)


def show_coverage_per_model(coverage):
    all_units = 0
    all_positive_ignitions = 0
    all_negative_ignitions = 0
    for c in coverage:
        p_ignitions, n_ignitions, units = c
        all_units += units
        all_positive_ignitions += p_ignitions
        all_negative_ignitions += n_ignitions

    p_rate = all_positive_ignitions / all_units
    n_rate = all_negative_ignitions / all_units
    print("positive ignition rate (per model): {:.2%}".format(p_rate))
    print("negative ignition rate (per model): {:.2%}".format(n_rate))
