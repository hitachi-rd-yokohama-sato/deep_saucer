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
import sys
import os
import tensorflow as tf
from pathlib import Path

_proj_dir = Path(__file__).absolute().parent.parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_dir = Path(_proj_dir, 'examples')
_examples_xor = Path(_examples_dir, 'xor')

sys.path.append(str(_proj_dir))
sys.path.append(str(_lib_dir))

os.chdir(str(Path(__file__).parent))
_current = Path(os.getcwd()).absolute()

from lib.dnn_encoding_verification import main

if __name__ == '__main__':
    sess = tf.Session()
    saver = tf.train.import_meta_graph(
        str(_examples_xor.joinpath('model', 'train.meta')))
    saver.restore(sess, str(_examples_xor.joinpath('model', 'train')))

    main(sess, config_path=str(_examples_xor.joinpath('configs', 'config_xor.json')))
