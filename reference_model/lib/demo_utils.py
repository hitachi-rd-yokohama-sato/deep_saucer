# -*- coding: utf-8 -*-
#******************************************************************************************
# Copyright (c) 2019 Hitachi, Ltd.
# All rights reserved. This program and the accompanying materials are made available under
# the terms of the MIT License which accompanies this distribution, and is available at
# https://opensource.org/licenses/mit-license.php
#
# March 1st, 2019 : First version.
#******************************************************************************************
import sys
import json
from pathlib import Path


def load_module(paths, func_names):
    version = sys.version_info
    major = version[0]
    minor = version[1]
    modules = []
    if major == 3 and 5 <= minor:
        import importlib.util
        for idx, args in enumerate(zip(paths, func_names)):
            path, func_name = args
            module_name = "module_{}".format(idx)
            spec = importlib.util.spec_from_file_location(module_name,
                                                          path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(getattr(module, func_name))
    elif major == 3 and minor == 3 or minor == 4:
        from importlib.machinery import SourceFileLoader
        for idx, args in enumerate(zip(paths, func_names)):
            path, func_name = args
            module_name = "module_{}".format(idx)
            spec = SourceFileLoader(module_name, path)
            module = spec.load_module()
            modules.append(getattr(module, func_name))
    else:
        raise EnvironmentError("{} not supported.".format(version))
    return modules


def is_key(data, keys):
    """
    :param dict data:
    :param list[str] keys:

    >>> query = {"hoge": 1, "huga": 2}
    >>> is_key(query, ["hoge", "huga"])
    >>> is_key(query, ["hoge", "huge"])
    Traceback (most recent call last):
    ...
    KeyError: 'huga'
    """
    for key in data.keys():
        if key not in keys:
            raise KeyError(key)


def check_path(path):
    """
    :param str path:
    >>> query_1 = ""
    >>> check_path(query_1)
    Traceback (most recent call last):
    ...
    ValueError
    >>> query_2 = "../hoge/hoge.csv"
    >>> check_path(query_2)
    Traceback (most recent call last):
    ...
    ValueError: ..\hoge\hoge.csv isn't abs_path.
    >>> query_3 = "config_gtsrb.json"
    >>> check_path(query_3)
    Traceback (most recent call last):
    ...
    FileExistsError: C:\python\demo_projects\configs\config_gtsrb.json
    """
    if not path:
        raise ValueError(path)
    path = Path(path)
    # OSARA's root_path undecided
    if not path.is_absolute():
        raise ValueError("{} isn't abs_path.".format(path))
    if not path.exists():
        raise FileExistsError(path)


def is_path(data):
    """
    :param dict data:
    """
    for key in data.keys():
        if key == "rdg_exe" or key == "function_names":
            paths = data[key]
            if not isinstance(paths, list):
                raise TypeError("{} not supported, only supported"
                                " type list".format(type(paths)))
            else:
                if key == "rdg_exe":
                    for path in paths:
                        check_path(path)
                else:
                    for path in paths:
                        if not path:
                            raise ValueError(path)
        else:
            path = data[key]
            check_path(path)


def load_config(json_file):
    """
    :param  str json_file:
    :rtype dict
    """
    with open(json_file, "r") as f:
        config = json.load(f)
    keys = ["rule", "rdg_exe", "rdg_weight", "rdg_output", "function_names"]
    is_key(config, keys)
    is_path(config)

    return config
