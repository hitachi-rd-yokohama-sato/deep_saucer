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
from tkinter import Frame, BOTH, YES, TOP, LEFT, BROWSE

from conf.configuration import (
    TEST_FUNC_HEADERS, MODEL_HEADERS, DATA_HEADERS,
    TREE_VIEW_SELECT_EVENT, ESCAPE_EVENT, VALUES, LINE_COLORS, TEST_FUNC,
    DATA_SCRIPT, MODEL_SCRIPT)
from src.com.common import get_select_value

from src.gui.list_frame import (
    TestFuncListFrame, ModelScriptListFrame, DataScriptListFrame, apply_env_id)
from src.info.data_script_info import DataScriptInfo
from src.info.test_func_info import TestFuncInfo
from src.info.model_script_info import ModelScriptInfo


class TestSelectFrame(Frame):

    def __init__(self, root=None):
        """
        init
        :param root:
        """
        self.__root = root

        Frame.__init__(self, self.__root)

        # List Box Frame
        self.__frame_tv = Frame(self)

        self.__test_func_list_frame = TestFuncListFrame(
            self.__frame_tv, select_mode=BROWSE, size=12)

        self.__model_script_list_frame = ModelScriptListFrame(
            self.__frame_tv, select_mode=BROWSE, size=12)

        self.__data_script_list_frame = DataScriptListFrame(
            self.__frame_tv, select_mode=BROWSE, size=12)

        self.__test_func_list_frame.pack(fill=BOTH, expand=YES, side=LEFT)
        self.__data_script_list_frame.pack(fill=BOTH, expand=YES, side=LEFT)
        self.__model_script_list_frame.pack(fill=BOTH, expand=YES, side=LEFT)

        self.__frame_tv.pack(fill=BOTH, expand=YES, side=TOP)

        # Verification List
        self.__test_func_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)
        self.__test_func_list_frame.bind_treeview(ESCAPE_EVENT, self.__escape)

        # Model Script List
        self.__model_script_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)
        self.__model_script_list_frame.bind_treeview(ESCAPE_EVENT,
                                                     self.__escape)
        # Processed Data List
        self.__data_script_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)
        self.__data_script_list_frame.bind_treeview(ESCAPE_EVENT, self.__escape)

    def __unbind(self):
        self.__test_func_list_frame.unbind_treeview(TREE_VIEW_SELECT_EVENT)
        self.__model_script_list_frame.unbind_treeview(TREE_VIEW_SELECT_EVENT)
        self.__data_script_list_frame.unbind_treeview(TREE_VIEW_SELECT_EVENT)

    def __bind(self):
        # set unbind event
        self.after(0, self.__tv_bind)

    def __tv_bind(self):
        self.__test_func_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)

        self.__model_script_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)

        self.__data_script_list_frame.bind_treeview(
            TREE_VIEW_SELECT_EVENT, self.__select_list)

    def update_list_items(self):
        self.__unbind()

        self.__test_func_list_frame.update_items(
            TEST_FUNC_HEADERS, TestFuncInfo.data_values())

        self.__model_script_list_frame.update_items(
            MODEL_HEADERS, ModelScriptInfo.data_values())

        self.__data_script_list_frame.update_items(
            DATA_HEADERS, DataScriptInfo.data_values())

        self.__bind()

    def __select_list(self, event):
        self.__unbind()

        select_item_id = event.widget.selection()[0]
        env_id = event.widget.item(select_item_id)[VALUES][1]
        self.__apply_env(env_id)

        self.__bind()

    def __apply_env(self, env_id):

        apply_env_id(self.__test_func_list_frame, env_id)
        apply_env_id(self.__data_script_list_frame, env_id)
        apply_env_id(self.__model_script_list_frame, env_id)

    def __escape(self, event):
        self.__unbind()

        tv = event.widget
        # select all clear
        for index, item_id in enumerate(tv.get_children('')):
            tv.selection_remove(item_id)
            tv.tag_configure(item_id, background=LINE_COLORS[index % 2])

        self.__bind()

    def get_select_test_func(self):
        return get_select_value(self.__test_func_list_frame, TestFuncInfo)

    def get_select_model(self):
        return get_select_value(self.__model_script_list_frame,
                                ModelScriptInfo)

    def get_select_data(self):
        return get_select_value(self.__data_script_list_frame, DataScriptInfo)

    def delete_selected(self):
        # Unbind event
        self.__unbind()
        del_dict = {}

        # Delete Verification Script
        # Delete GUI
        del_list = self.__test_func_list_frame.delete_selection()
        del_dict[TEST_FUNC] = [
            (TEST_FUNC, v, TestFuncInfo.get_data(v).abs_path)
            for v in del_list]
        # Delete Data
        TestFuncInfo.delete_data(del_list)

        # Delete Dataset Load Script
        # Delete GUI
        del_list = self.__data_script_list_frame.delete_selection()
        del_dict[DATA_SCRIPT] = [
            (DATA_SCRIPT, v, DataScriptInfo.get_data(v).abs_path)
            for v in del_list]
        # Delete DATA
        DataScriptInfo.delete_data(del_list)

        # Delete Model Load Script
        # Delete GUI
        del_list = self.__model_script_list_frame.delete_selection()
        del_dict[MODEL_SCRIPT] = [
            (MODEL_SCRIPT, v, ModelScriptInfo.get_data(v).abs_path)
            for v in del_list]
        # Delete DATA
        ModelScriptInfo.delete_data(del_list)

        # Bind event
        self.__bind()

        # Update view
        self.update_list_items()

        return del_dict
