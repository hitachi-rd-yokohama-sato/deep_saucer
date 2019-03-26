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
import os
from pathlib import PurePath, Path

from conf.configuration import HOME


class BaseData(object):

    def __init__(self, identifier, path):
        self._path = path
        self._id = identifier

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @id.deleter
    def id(self):
        del self._id

    @property
    def name(self):
        return os.path.basename(self._path)

    @property
    def path(self):
        return self._path

    @path.deleter
    def path(self):
        del self._path

    @property
    def abs_path(self):
        return os.path.join(self.dir_path, self.name)

    @property
    def data_tuple(self):
        return self._id, self.name, self.abs_path

    @property
    def dir_path(self):
        p = PurePath(self._path)
        if not p.is_absolute():
            p = PurePath(HOME).joinpath(self._path)
        return str(Path(p).parent.resolve())
