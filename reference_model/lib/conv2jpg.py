#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import shutil

from scipy import misc
from pathlib import Path

from keras.utils import HDF5Matrix


def convert_from_hdf5():
    cur_dir = Path(__file__).parent.absolute()

    # examples/gtsrb/data/test_dataset_3_32_32.h5
    input_path = cur_dir.parent.joinpath(
        'examples', 'gtsrb', 'data', 'test_dataset_3_32_32.h5')

    # output of pngs and textfiles
    output_folder = cur_dir.joinpath('html', 'images')
    if output_folder.exists():
        shutil.rmtree(str(output_folder))

    output_folder.mkdir(exist_ok=True)

    print('Input: %s' % input_path)
    print('Output: %s' % output_folder)

    convert_images_from_hdf5(input_path, output_folder)
    print('Done')


def convert_images_from_hdf5(input_path, output_folder):
    images = HDF5Matrix(input_path, 'images')

    print('Converting images')
    for i, image in enumerate(images):
        img = image.transpose((1, 2, 0))
        misc.imsave(output_folder.joinpath('%d.jpg' % i), img)


if __name__ == '__main__':
    convert_from_hdf5()
