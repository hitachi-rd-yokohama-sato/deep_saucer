# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 :
# Derived from gen_diff.py
#******************************************************************************************

from __future__ import print_function

import sys
from scipy.misc import imsave

from configs import bcolors
from utils import *

import json
import os
_deepxplore_dir = os.path.dirname(os.path.dirname(__file__))
_deepxplore_mnist_dir = os.path.join(_deepxplore_dir, 'MNIST')

sys.path.append(_deepxplore_dir)
sys.path.append(_deepxplore_mnist_dir)


TRANSFORMATION = 'transformation'
WEIGHT_DIFF = 'weight_diff'
WEIGHT_NC = 'weight_nc'
STEP = 'step'
SEEDS = 'seeds'
GRAD_ITERATIONS = 'grad_iterations'
THRESHOLD = 'threshold'
TARGET_MODEL = 'target_model'
START_POINT = 'start_point'
OCCLUSION_SIZE = 'occlusion_size'


class Args(object):
    """
    Used instead of ArgumentParser
    """
    def __init__(self):
        self.args_dic = {}
        self.options = [TRANSFORMATION, WEIGHT_DIFF, WEIGHT_NC, STEP, SEEDS,
                        GRAD_ITERATIONS, THRESHOLD, TARGET_MODEL, START_POINT,
                        OCCLUSION_SIZE]

    @property
    def transformation(self):
        return self.args_dic[TRANSFORMATION]

    @property
    def weight_diff(self):
        return self.args_dic[WEIGHT_DIFF]

    @property
    def weight_nc(self):
        return self.args_dic[WEIGHT_NC]

    @property
    def step(self):
        return self.args_dic[STEP]

    @property
    def seeds(self):
        return self.args_dic[SEEDS]

    @property
    def grad_iterations(self):
        return self.args_dic[GRAD_ITERATIONS]

    @property
    def threshold(self):
        return self.args_dic[THRESHOLD]

    @property
    def target_model(self):
        return self.args_dic[TARGET_MODEL]

    @property
    def start_point(self):
        return self.args_dic[START_POINT]

    @property
    def occlusion_size(self):
        return self.args_dic[OCCLUSION_SIZE]

    def load_args(self, j_path):
        """
        Load the value set in the JSON file
        :param j_path:
        :return:
        """
        with open(j_path) as fs:
            j_data = json.load(fs)

        # Check that the required options are set
        # Set values for unset options
        for arg_name in self.options:
            if arg_name == TRANSFORMATION:
                # args.transformation
                val = _get_str(j_data, arg_name,
                               range_val=['light', 'occl', 'blackout'])

                self.args_dic[arg_name] = val

            elif arg_name in [WEIGHT_DIFF, WEIGHT_NC, STEP, THRESHOLD]:
                # args.weight_diff
                # args.weight_nc
                # args.step
                # args.threshold
                val = _get_float(j_data, arg_name)
                self.args_dic[arg_name] = val

            elif arg_name in [SEEDS, GRAD_ITERATIONS]:
                # args.seeds
                # args.grad_iterations
                val = _get_int(j_data, arg_name)
                self.args_dic[arg_name] = val

            elif arg_name == TARGET_MODEL:
                # args.target_model
                val = _get_int(j_data, arg_name, default=0,
                               range_val=range(3))
                self.args_dic[arg_name] = val

            elif arg_name == START_POINT:
                # args.start_point
                val = _get_num_tuple(j_data, arg_name, default=(0, 0))
                self.args_dic[arg_name] = val

            elif arg_name == OCCLUSION_SIZE:
                # args.occlusion_size
                val = _get_num_tuple(j_data, arg_name, default=(10, 10))
                self.args_dic[arg_name] = val

        if any(x is None for x in self.args_dic.values()):
            return False

        return True


def _get_str(j_data, key, default=None, range_val=None):
    """
    Get data as str
    :param j_data: Result of loading JSON
    :param key: The value key to retrieve
    :param default: Default value if not set
    :param range_val: Range of values that can be set
    :return:
    """
    value = j_data.get(key, default)
    if value is None:
        sys.stderr.write('"%s" is required\n' % key)
        return None

    if not isinstance(value, unicode):
        sys.stderr.write('"%s" choose from %s\n' % (key, range_val))
        return None

    if value not in range_val:
        sys.stderr.write('"%s" choose from %s\n' % (key, range_val))
        return None

    return value


def _get_float(j_data, key, default=None, range_val=None):
    """
    Get data as float
    :param j_data: Result of loading JSON
    :param key: The value key to retrieve
    :param default: Default value if not set
    :param range_val: Range of values that can be set
    :return:
    """
    value = j_data.get(key, default)
    if value is None:
        sys.stderr.write('"%s" is required\n' % key)
        return value

    elif _is_number(value):
        if range_val and value not in range_val:
            sys.stderr.write('"%s" choose from %s\n' % (key, range_val))
            return None

        return float(value)

    else:
        sys.stderr.write('"%s" set a numerical value\n' % key)
        return None


def _get_int(j_data, key, default=None, range_val=None):
    """
    Get data as int
    :param j_data: Result of loading JSON
    :param key: The value key to retrieve
    :param default: Default value if not set
    :param range_val: Range of values that can be set
    :return:
    """
    value = j_data.get(key, default)
    if value is None:
        sys.stderr.write('"%s" is required\n' % key)
        return value

    elif _is_number(value):
        if range_val and value not in range_val:
            sys.stderr.write('"%s" choose from %s\n' % (key, range_val))
            return None

        return int(value)

    else:
        sys.stderr.write('"%s" set a integer value\n' % key)
        return None


def _get_num_tuple(j_data, key, default=None):
    """
    Get data as number tuple
    :param j_data: Result of loading JSON
    :param key: The value key to retrieve
    :param default: Default value if not set
    :return:
    """
    value = j_data.get(key, default)
    if isinstance(value, list):
        value = tuple(value)

    if not isinstance(value, tuple):
        sys.stderr.write('"%s" is a list of integer values\n' % key)
        return None
    else:
        for v in value:
            if not _is_number(v):
                sys.stderr.write('"%s" is a list of integer values\n' % key)
                return None

    return value


def _is_number(val):
    """
    Determine if the value is a number
    :param val:
    :return:
    """
    if isinstance(val, list):
        for v in val:
            if not _is_number(v):
                return False

        return True

    return isinstance(val, int) or isinstance(val, float)


def main(models, dataset=None, config_path=None):
    """
    deepxplore/MNIST/gen_digg.py(custom)
    :param dataset: test_images
    :param models: List containing three models and keras.layers.Input()
    :param config_path: JSON file path with parameters set
    :return:
    """
    # input image dimensions
    img_rows, img_cols = 28, 28
    # Get parameters from json file
    # Set Args class
    args = Args()
    if not args.load_args(config_path):
        raise Exception

    # input images
    x_test = dataset
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255

    # input_tensor is also included in the result of Model Load Script
    if len(models) != 4:
        sys.stderr.write('The result of ModelLoadScript must be '
                         'a list containing three models and '
                         'keras.layers.Input()\n')
        raise Exception

    # define input tensor as a placeholder
    input_tensor = models[3]

    # load multiple models sharing same input tensor
    model1 = models[0]
    model2 = models[1]
    model3 = models[2]

    # init coverage table
    model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

    # ==============================================================================================
    # start gen inputs
    for _ in xrange(args.seeds):
        gen_img = np.expand_dims(random.choice(x_test), axis=0)
        orig_img = gen_img.copy()
        # first check if input already induces differences
        label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
            model3.predict(gen_img)[0])

        if not label1 == label2 == label3:
            print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2,
                                                                                                label3) + bcolors.ENDC)

            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                   neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                   neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)

            # save the result to disk
            imsave('./generated_inputs/' + 'already_differ_' + str(label1) + '_' + str(
                label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
            continue

        # if all label agrees
        orig_label = label1
        layer_name1, index1 = neuron_to_cover(model_layer_dict1)
        layer_name2, index2 = neuron_to_cover(model_layer_dict2)
        layer_name3, index3 = neuron_to_cover(model_layer_dict3)

        # construct joint loss function
        if args.target_model == 0:
            loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        elif args.target_model == 1:
            loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        elif args.target_model == 2:
            loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
        loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
        loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
        layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

        # we run gradient ascent for 20 steps
        for iters in xrange(args.grad_iterations):
            loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
                [gen_img])
            if args.transformation == 'light':
                grads_value = constraint_light(grads_value)  # constraint the gradients value
            elif args.transformation == 'occl':
                grads_value = constraint_occl(grads_value, args.start_point,
                                              args.occlusion_size)  # constraint the gradients value
            elif args.transformation == 'blackout':
                grads_value = constraint_black(grads_value)  # constraint the gradients value

            gen_img += grads_value * args.step
            predictions1 = np.argmax(model1.predict(gen_img)[0])
            predictions2 = np.argmax(model2.predict(gen_img)[0])
            predictions3 = np.argmax(model3.predict(gen_img)[0])

            if not predictions1 == predictions2 == predictions3:
                update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
                update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
                update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

                print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                      % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                         neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                         neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
                averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                               neuron_covered(model_layer_dict3)[0]) / float(
                    neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                    neuron_covered(model_layer_dict3)[
                        1])
                print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

                gen_img_deprocessed = deprocess_image(gen_img)
                orig_img_deprocessed = deprocess_image(orig_img)

                # save the result to disk
                imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                    predictions2) + '_' + str(predictions3) + '.png',
                       gen_img_deprocessed)
                imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                    predictions2) + '_' + str(predictions3) + '_orig.png',
                       orig_img_deprocessed)
                break
