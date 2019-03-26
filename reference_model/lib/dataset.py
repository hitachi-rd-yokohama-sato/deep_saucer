# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import h5py
import numpy as np
import pandas as pd
from keras.utils import HDF5Matrix, to_categorical
from pathlib import Path
from skimage import color, exposure, transform, io

Training_PATH = Path(r'GTSRB_data/Final_Training/Images')
Test_PATH = Path(r'GTSRB_data/Final_Test/Images')
Test_CSV_Path = Path(r'GTSRB_data/GT-final_test.csv')


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


def run_image(img, rgb=True, width=32, height=32):
    # Histogram normalization in v channel only
    # rgb add Lightness
    """
    :type width: int
    :type height: int
    :type rgb: True
    :type img: np.ndarray
    :rtype: np.ndarray
    """
    # central square crop
    # get minimum shape height or width
    centre = img.shape[0] // 2, img.shape[1] // 2

    if rgb:
        img = cnv2rgb(img, centre, width, height)
    else:
        img = cnv2gs(img, centre, width, height)
    return img


def cnv2gs(img, centre, width, height):
    img = color.rgb2gray(img)
    min_side = min(img.shape)
    img = img[
          centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2]
    # rescale to standard size
    img = transform.resize(img, (width, height))

    return img


def cnv2rgb(img, centre, width, height):
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    min_side = min(img.shape[:-1])
    img = img[
          centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]
    # rescale to standard size
    img = transform.resize(img, (width, height))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)

    return img


def get_image_paths(root_path_dir, shuffle=False, regex="*/*.ppm"):
    """
    :type root_path_dir: pathlib.Path
    :type shuffle: bool
    :type regex: str
    :rtype: list[pathlib.Path]

    ex.
    root - dir -- image1.jpg
               -- image2.jpg
               -- ...
    root_path_dir <-- root/dir
    """
    all_img_paths = root_path_dir.glob(regex)
    all_img_paths = list(all_img_paths)

    if shuffle:
        np.random.shuffle(all_img_paths)

    return all_img_paths


def preprocess(file_name, classes=43, width=32, height=32, rgb=True):
    test = pd.read_csv(Test_CSV_Path, sep=';')
    file_names = test['Filename']
    class_ids = test['ClassId']
    samples = len(class_ids)
    labels = np.empty((samples, classes))
    if rgb:
        channels = 3
        images = np.empty((samples, channels, width, height))
        for idx, comp in enumerate(zip(file_names, class_ids)):
            path, class_id = comp
            img = io.imread(str(Test_PATH.joinpath(path)))
            images[idx] = run_image(img)
            labels[idx] = to_categorical(class_id, classes)
    else:
        images = np.empty((samples, width, height))
        for idx, comp in enumerate(zip(file_names, class_ids)):
            path, class_id = comp
            img = io.imread(str(Test_PATH.joinpath(path)))
            images[idx] = run_image(img, rgb=rgb)
            labels[idx] = to_categorical(class_id, classes)
    with h5py.File(file_name, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)
    return images


def data_create(root_path):
    """
    Notice that it is a calling function dedicated to OSARA Tool.

    Load HDF5 format file and dataset name is "images".

    !Required: Function Name is "data_create"

    :type root_path: str
    :rtype: numpy.ndarray

    >>> root_path = r"C:\cygwin64\home\spc000\python\\new_project\input_data"
    >>> dataset = data_create(root_path)
    """
    root_path = Path(root_path)
    file_name = "test_dataset_3_32_32.h5"
    dataset_path = root_path.joinpath(file_name)
    if dataset_path.exists():
        test_x = HDF5Matrix(dataset_path, "images")
    else:
        test_x = preprocess(dataset_path)
    return test_x
