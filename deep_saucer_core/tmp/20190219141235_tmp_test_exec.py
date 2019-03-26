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
import time


if __name__ == '__main__':
    args = sys.argv

    sys.path.append(r'/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/xor/model')
    sys.path.append(r'/home/sato/PycharmProjects/deepsaucer_publish/dnn_encoding/lib')

    print('[INFO] Import Script')
    sys.stdout.flush()
    import model_dnn_encoding as ml
    import dnn_encoding_verification as tf

    # ## CALL_DATA_LOADER ## #

    print('[INFO] Call Model Load Script')
    sys.stdout.flush()
    model = ml.model_load('/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/downloaded_data')
    sys.stdout.flush()

    if model is None:
        sys.stderr.write('Failed to acquire model')
        sys.exit(1)

    print('[INFO] Call Verification Script')
    sys.stdout.flush()
    time.sleep(1)
    result = tf.main(model, None, "/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/xor/configs/config_dnn_encoding.json")

    sys.exit(0)
