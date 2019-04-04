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
from tkinter import (
    Frame, Toplevel, Button,
    ACTIVE, DISABLED, EXTENDED, BOTH, NE, NW, YES, LEFT, RIGHT, TOP
)
from conf.configuration import (
    ENV_SETUP, ENV_SETUP_HEADERS, TREE_VIEW_SELECT_EVENT,
    ADD, DEL, LINKED_ENV, NON_LINKED_ENV
)
from src.com.common import get_geometry
from src.gui.list_frame import (
    TestFuncListFrame, ModelScriptListFrame, DataScriptListFrame, BaseListFrame
)
from src.info.data_script_info import DataScriptInfo
from src.info.test_func_info import TestFuncInfo
from src.info.model_script_info import ModelScriptInfo
from src.info.env_setup_info import EnvSetupInfo


class EnvEditWizard(Toplevel):
    """
    Linked Env Script Edit Wizard
    """
    def __init__(self, master=None, list_frame=None, item_id=None,
                 select_mode=EXTENDED, width=0.45, height=0.45, size=10,
                 use_factor=True):

        Toplevel.__init__(self, master=master)

        w, h, x, y = get_geometry(self, width, height, use_factor)

        self.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.list_frame = list_frame
        self.item_id = item_id

        if isinstance(self.list_frame, TestFuncListFrame):
            # select Verification script
            identifier = self.list_frame.get_item_value(self.item_id, 0)
            self.__script = TestFuncInfo.get_data(identifier)
        elif isinstance(self.list_frame, ModelScriptListFrame):
            # select Model script
            identifier = self.list_frame.get_item_value(self.item_id, 0)
            self.__script = ModelScriptInfo.get_data(identifier)
        elif isinstance(self.list_frame, DataScriptListFrame):
            # select Data script
            identifier = self.list_frame.get_item_value(self.item_id, 0)
            self.__script = DataScriptInfo.get_data(identifier)
        else:
            return

        # Linked env id
        self.__linked_env = [
            item for item in EnvSetupInfo.data_values()
            if item.id in self.__script.env_id]

        # Non-linked env id
        self.__non_linked_env = [
            item for item in EnvSetupInfo.data_values()
            if item.id not in self.__script.env_id]

        # Linked Env list Frame
        linked_frame = Frame(self)
        self.__linked_env_list_frame = EnvListFrame(
            linked_frame, LINKED_ENV, self.__linked_env,
            select_mode=select_mode, size=size)
        # Delete button select command
        self.__del_btn = Button(linked_frame, text=DEL, width=6,
                                command=self.__del_env)
        
        self.__linked_env_list_frame.pack(fill=BOTH, anchor=NW,
                                          expand=YES, side=TOP)
        self.__del_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)

        # Non-linked Env list Frame
        non_linked_frame = Frame(self)
        self.__non_linked_env_list_frame = EnvListFrame(
            non_linked_frame, NON_LINKED_ENV, self.__non_linked_env,
            select_mode=select_mode, size=size)
        # Add button select command
        self.__add_btn = Button(non_linked_frame, text=ADD, width=6,
                                command=self.__add_env)

        self.__non_linked_env_list_frame.pack(fill=BOTH, anchor=NW,
                                              expand=YES, side=TOP)
        self.__add_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)

        linked_frame.pack(fill=BOTH, expand=YES, side=LEFT)
        non_linked_frame.pack(fill=BOTH, expand=YES, side=LEFT)

        if len(self.__linked_env) == 1:
            self.__del_btn.configure(state=DISABLED)

    def __add_env(self):
        for item_id in self.__non_linked_env_list_frame.get_selection():
            # add env
            env_id = self.__non_linked_env_list_frame.get_item_value(
                item_id, 0)
            e_script = [e for e in self.__non_linked_env if env_id == e.id][0]

            # add linked env list
            self.__linked_env.append(e_script)
            # remove non linked env list
            self.__non_linked_env.remove(e_script)

        # sort list
        self.__linked_env = sorted(self.__linked_env, key=lambda e: e.id)
        self.__non_linked_env = sorted(
            self.__non_linked_env, key=lambda e: e.id)

        # update list view
        self.__linked_env_list_frame.update_items(
            ENV_SETUP_HEADERS, self.__linked_env)
        self.__non_linked_env_list_frame.update_items(
            ENV_SETUP_HEADERS, self.__non_linked_env)

        # update script env info
        self.__script.env_id = [e.id for e in self.__linked_env]

        if len(self.__linked_env) > 1:
            self.__del_btn.configure(state=ACTIVE)

    def __del_env(self):
        for item_id in self.__linked_env_list_frame.get_selection():
            # del env
            env_id = self.__linked_env_list_frame.get_item_value(
                item_id, 0)
            e_script = [e for e in self.__linked_env if env_id == e.id][0]

            # add non linked env list
            self.__non_linked_env.append(e_script)
            # remove linked env list
            self.__linked_env.remove(e_script)

        # sort list
        self.__linked_env = sorted(self.__linked_env,
                                       key=lambda e: e.id)
        self.__non_linked_env = sorted(
            self.__non_linked_env, key=lambda e: e.id)

        # update list view
        self.__linked_env_list_frame.update_items(
            ENV_SETUP_HEADERS, self.__linked_env)
        self.__non_linked_env_list_frame.update_items(
            ENV_SETUP_HEADERS, self.__non_linked_env)

        # update script env info
        self.__script.env_id = [e.id for e in self.__linked_env]

        if len(self.__linked_env) == 1:
            self.__del_btn.configure(state=DISABLED)


class EnvListFrame(BaseListFrame):
    def __init__(self, root=None, label='', values=None,
                 select_mode=EXTENDED, size=10):
        BaseListFrame.__init__(self, root=root, label=label,
                               columns=ENV_SETUP_HEADERS, values=values,
                               select_mode=select_mode, size=size)
