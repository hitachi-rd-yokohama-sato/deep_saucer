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

    sys.path.append(r'/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/xor/data')
    sys.path.append(r'/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/xor/model')
    sys.path.append(r'/home/sato/PycharmProjects/deepsaucer_publish/assertion_testing/lib')

    print('[INFO] Import Script')
    sys.stdout.flush()
    import dataset_assertion as dc
    import model_assertion_check as ml
    import assertion_check_verification as tf

    print('[INFO] Call Dataset Load Script')
    sys.stdout.flush()
    data = dc.data_create('/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/downloaded_data')
    sys.stdout.flush()

    if data is None:
        sys.stderr.write('Failed to acquire data')
        sys.exit(1)

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
    result = tf.main(model, data, "/home/sato/PycharmProjects/deepsaucer_publish/deep_saucer_core/xor/configs/config_assertion_check.json")

    sys.exit(0)
