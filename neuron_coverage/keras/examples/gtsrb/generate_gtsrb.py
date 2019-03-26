# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019
# School of Electronics and Computer Science, University of Southampton and Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import argparse
import h5py
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from pathlib import Path
from skimage import color, exposure, transform, io


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
    :type height: int
    :type width: int
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


def preprocess_test(file_name, classes=43, width=32, height=32, rgb=True):
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
            images[idx] = run_image(img, width=width, height=height)
            labels[idx] = to_categorical(class_id, classes)

            if idx != 0 and idx % 10000 == 0:
                print(idx)

    else:
        images = np.empty((samples, width, height))
        for idx, comp in enumerate(zip(file_names, class_ids)):
            path, class_id = comp
            img = io.imread(str(Test_PATH.joinpath(path)))
            images[idx] = run_image(img, width=width, height=height, rgb=rgb)
            labels[idx] = to_categorical(class_id, classes)

            if idx != 0 and idx % 10000 == 0:
                print(idx)

    with h5py.File(file_name, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)
    return images


def preprocess_train(file_name, classes=43, width=32, height=32, rgb=True):
    train_image_paths = get_image_paths(Training_PATH, shuffle=True)
    samples = len(train_image_paths)
    labels = np.empty((samples, classes))
    if rgb:
        channels = 3
        images = np.empty((samples, channels, width, height))
        for idx, path in enumerate(train_image_paths):
            class_id = int(path.parent.name)
            img = io.imread(str(path))
            images[idx] = run_image(img, width=width, height=height)
            labels[idx] = to_categorical(class_id, classes)

            if idx != 0 and idx % 10000 == 0:
                print(idx)

    else:
        images = np.empty((samples, width, height))
        for idx, path in enumerate(train_image_paths):
            class_id = int(path.parent.name)
            img = io.imread(str(path))
            images[idx] = run_image(img, width=width, height=height, rgb=rgb)
            labels[idx] = to_categorical(class_id, classes)

            if idx != 0 and idx % 10000 == 0:
                print(idx)

    with h5py.File(file_name, "w") as hf:
        hf.create_dataset("images", data=images)
        hf.create_dataset("labels", data=labels)
    return images


def main(args):
    """
    train_dataset_C_W_H.h5 -> train data with images and labels
    test_dataset_C_W_H.h5  -> train data with images and labels
　　images format (image-num, channels, height, width)
　　labels format (image-num, classes)
    """
    if args.rgb:
        name_label = '3_{0}_{1}'.format(args.width, args.height)
    else:
        name_label = '1_{0}_{1}'.format(args.width, args.height)
    train_file_name = "train_dataset_{}.h5".format(name_label)
    train_save_path = Output_PATH.joinpath(train_file_name).absolute()

    test_file_name = "test_dataset_{}.h5".format(name_label)
    test_save_path = Output_PATH.joinpath(test_file_name).absolute()

    print('Generating Train Data...')
    preprocess_train(file_name=str(train_save_path),
                     classes=args.classes, width=args.width, height=args.height,
                     rgb=args.rgb)
    print(str(train_save_path))

    print('Generating Test Data...')
    preprocess_test(file_name=str(test_save_path),
                    classes=args.classes, width=args.width, height=args.height,
                    rgb=args.rgb)
    print(str(test_save_path))


if __name__ == '__main__':

    _arg_parser = argparse.ArgumentParser()
    _arg_parser.add_argument('-d', '--dir', type=str, required=True)
    _arg_parser.add_argument('--classes', type=int, default=43)
    _arg_parser.add_argument('--width', type=int, default=32)
    _arg_parser.add_argument('--height', type=int, default=32)
    _arg_parser.add_argument('-o', '--out_dir')
    _arg_parser.add_argument('--rgb', type=bool, default=True)

    _args = _arg_parser.parse_args()

    GTSRB_data_PATH = Path(_args.dir)
    Training_PATH = GTSRB_data_PATH.joinpath('Final_Training', 'Images')
    Test_PATH = GTSRB_data_PATH.joinpath('Final_Test', 'Images')
    Test_CSV_Path = GTSRB_data_PATH.joinpath('GT-final_test.csv')

    if _args.out_dir is None:
        Output_PATH = Path(__file__).parent
    else:
        Output_PATH = Path(_args.out_dir)

    main(_args)
