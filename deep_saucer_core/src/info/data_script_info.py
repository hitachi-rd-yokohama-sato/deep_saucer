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

from conf.configuration import DATA_SCRIPT, ID, PATH, UTF8, ENV_SETUP
from src.info.base_info import BaseData


class DataScriptInfo(object):
    __data_dict = {}
    __data_script_dict = {}

    @classmethod
    def read_conf(cls, path):
        try:
            with open(file=path, mode='r', encoding=UTF8) as read_file:
                load_val = yaml.load(read_file)
                if load_val and DATA_SCRIPT in load_val:
                        for val in load_val[DATA_SCRIPT]:
                            data_script = DataScript(identifier=val[ID],
                                                     path=val[PATH],
                                                     env_id=val[ENV_SETUP])

                            cls.add_data(data_script)

        except Exception as e:
            print(e)
            return False

        return True

    @classmethod
    def data(cls):
        data_list = []

        for identifier, data in cls.data_items():
            data_list.append(OrderedDict(
                {ID: data.id, PATH: data.path, ENV_SETUP: data.env_id}))

        return {DATA_SCRIPT: data_list}

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
    def get_data_with_path_eid(cls, path, env_id):
        values = [value for value in cls.data_values() if
                  value.abs_path == path and value.env_id == env_id]
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

    @classmethod
    def delete_data(cls, index):
        if type(index) is not list:
            if index is cls.max_id():
                del cls.__data_dict[index]

            else:
                for i in range(len(cls.data_items()) - 1):
                    if i < index:
                        continue

                    cls.__data_dict[i] = cls.__data_dict[i + 1]
                    cls.__data_dict[i].id = i
                del cls.__data_dict[cls.max_id()]
        else:
            for i in index:
                cls.delete_data(i)


class DataScript(BaseData):
    def __init__(self, path, env_id, identifier=None):
        if identifier is None:
            identifier = DataScriptInfo.max_id() + 1

        BaseData.__init__(self, identifier=identifier, path=path)

        self.__env_id = env_id

    @property
    def data_tuple(self):
        return self._id, self.__env_id, self.name, self.abs_path

    @property
    def env_id(self):
        return self.__env_id

    @env_id.deleter
    def env_id(self):
        del self.__env_id
