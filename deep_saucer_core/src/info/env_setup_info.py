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
from collections import OrderedDict

import yaml

from conf.configuration import ENV_SETUP, ID, PATH, UTF8
from src.info.base_info import BaseData


class EnvSetupInfo(object):
    __data_dict = {}

    @classmethod
    def read_conf(cls, path):
        try:
            with open(file=path, mode='r', encoding=UTF8) as read_file:
                load_val = yaml.load(read_file)
                if load_val and ENV_SETUP in load_val:
                    for val in load_val[ENV_SETUP]:
                        env_setup = EnvSetup(identifier=val[ID],
                                             path=val[PATH])
                        cls.add_data(env_setup)
        except Exception as e:
            print(e)
            return False

        return True

    @classmethod
    def data(cls):
        data_list = []
        for identifier, env_setup in cls.data_items():
            data_list.append(OrderedDict(
                {ID: env_setup.id, PATH: env_setup.path}))

        return {ENV_SETUP: data_list}

    @classmethod
    def data_items(cls):
        return cls.__data_dict.items()

    @classmethod
    def data_values(cls):
        return cls.__data_dict.values()

    @classmethod
    def max_id(cls):
        if len(cls.__data_dict) == 0:
            return -1
        return max(cls.__data_dict.keys())

    @classmethod
    def get_data(cls, key):
        if key not in cls.__data_dict.keys():
            return None
        else:
            return cls.__data_dict[key]

    @classmethod
    def get_data_with_path(cls, path):
        values = [value for value in cls.data_values() if
                  value.abs_path == path]
        if len(values) > 0:
            return values[0]
        else:
            return None

    @classmethod
    def add_data(cls, value):
        if value.id not in cls.__data_dict.keys():
            cls.__data_dict[value.id] = value
            return True
        else:
            return False


class EnvSetup(BaseData):

    def __init__(self, path, identifier=None):
        if identifier is None:
            identifier = EnvSetupInfo.max_id() + 1

        BaseData.__init__(self, identifier, path)
