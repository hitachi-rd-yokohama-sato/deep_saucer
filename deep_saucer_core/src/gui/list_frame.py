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
    Frame, EXTENDED, Label, NW, TOP, ttk, Scrollbar, VERTICAL,
    BOTH, RIGHT, HORIZONTAL, BOTTOM, YES, CENTER, END)
from tkinter.font import Font, BOLD

from conf.configuration import (
    LINE_COLORS, DISABLE_COLOR,
    TEST_FUNC_HEADERS, MODEL_HEADERS, DATA_HEADERS,
    FONT_NAME,  HEADINGS, VALUES,
    BACK_GROUND, TEST_FUNC_LABEL, MODEL_SCRIPT_LABEL, DATA_SCRIPT_LABEL)

from src.info.data_script_info import DataScriptInfo
from src.info.test_func_info import TestFuncInfo
from src.info.model_script_info import ModelScriptInfo


class BaseListFrame(Frame):

    def __init__(self, root=None, label='', size=10, columns=None, values=None,
                 select_mode=EXTENDED):
        self.__root = root
        Frame.__init__(self, self.__root)

        if columns is None:
            columns = []
        if values is None:
            values = []

        frame_tv = Frame(self)
        self.__f_size = size

        self._label = Label(frame_tv, text=label,
                            font=(FONT_NAME, size, BOLD))
        self._label.pack(anchor=NW, side=TOP, padx=5)

        self._tv = ttk.Treeview(frame_tv, columns=columns, show=HEADINGS,
                                height=5, selectmode=select_mode)

        self.insert_data(columns=columns, values=values)
        # Width adjustment
        for index, column in enumerate(columns):
            self._tv.column(index, width=Font().measure(column))

        sb_y = Scrollbar(frame_tv, orient=VERTICAL, command=self._tv.yview)
        self._tv.configure(yscrollcommand=sb_y.set)
        sb_y.pack(fill=BOTH, side=RIGHT, pady=5)

        sb_x = Scrollbar(frame_tv, orient=HORIZONTAL, command=self._tv.xview)
        self._tv.configure(xscrollcommand=sb_x.set)
        sb_x.pack(fill=BOTH, side=BOTTOM, padx=5)

        frame_tv.pack(fill=BOTH, anchor=NW, side=TOP, expand=YES, padx=5,
                      pady=5)
        self._tv.pack(fill=BOTH, expand=YES, side=TOP)

    def set_bg_color(self, item_id, bg):
        self._tv.tag_configure(item_id, background=bg,
                               font=(FONT_NAME, self.__f_size))

    def get_bg_color(self, item_id):
        return self._tv.tag_configure(item_id, BACK_GROUND)

    def insert_data(self, columns, values):
        for item_id in self._tv.get_children():
            self._tv.delete(item_id)
        # headers
        for index, column in enumerate(columns):
            self._tv.heading(index, text=column, anchor=CENTER,
                             command=lambda col=index:
                             self.sort_by(col, descending=False))

        # insert data
        for index, value in enumerate(values):
            item_id = self._tv.insert('', END, values=value.data_tuple)
            self._tv.item(item_id, tags=item_id)
            self.set_bg_color(item_id, LINE_COLORS[index % 2])

    def get_items(self):
        return self._tv.get_children('')

    def update_items(self, columns, values):
        self.clear_tv_select()
        self.insert_data(columns=columns, values=values)

    def add_data(self, value):
        index = len(self._tv.get_children(''))

        item_id = self._tv.insert('', END, values=value)
        self._tv.item(item_id, tags=item_id)
        self.set_bg_color(item_id, LINE_COLORS[index % 2])

    def get_item_value(self, item_id, index=0):
        return self._tv.item(item_id)[VALUES][index]

    def bind_treeview(self, event, command):
        self._tv.bind(event, command)

    def unbind_treeview(self, event):
        self._tv.bind(event, lambda e: 'break')

    def init_bg_color(self):
        for index, item_id in enumerate(self.get_items()):
            self.set_bg_color(item_id, LINE_COLORS[index % 2])

    def clear_tv_select(self):
        for index, item_id in enumerate(self.get_items()):
            # deselection
            self.selection_remove(item_id)
            # init background color
            self.set_bg_color(item_id, LINE_COLORS[index % 2])

    def get_selection(self):
        return self._tv.selection()

    def selection_remove(self, item_id):
        self._tv.selection_remove(item_id)

    def delete_selection(self):
        result = []
        for item_id in self.get_selection():
            identifier = self.get_item_value(item_id, 0)
            self._tv.delete(item_id)
            result.append(identifier)

        return result

    def sort_by(self, col, descending=False):
        # save background
        list_of_items = self._tv.get_children('')
        bg_colors = {}
        for item_id in list_of_items:
            bg_color = self.get_bg_color(item_id)
            if str(bg_color) == DISABLE_COLOR:
                bg_colors[item_id] = bg_color

        data = [(self._tv.set(child_id, col), child_id) for child_id in
                self._tv.get_children('')]
        # if the data to be sorted is numeric change to float
        try:
            data = [(float(number), child_ID) for number, child_ID in data]
        except ValueError:
            pass

        # now sort the data in place
        data.sort(reverse=descending)
        for idx, item in enumerate(data):
            self._tv.move(item[1], '', idx)

        # switch the heading so that it will sort in the opposite direction
        self._tv.heading(col,
                         command=lambda c=col: self.sort_by(c, not descending))

        # init color
        self.init_bg_color()

        # DISABLE_COLOR
        for item_id, bg_color in bg_colors.items():
            self.set_bg_color(item_id, bg_color)


def apply_env_id(list_frame, env_id):
    list_frame.init_bg_color()
    list_of_items = list_frame.get_items()

    for item_id in list_of_items:
        if list_frame.get_item_value(item_id, 1) is not env_id:
            list_frame.set_bg_color(item_id, DISABLE_COLOR)

    for item_id in list_frame.get_selection():
        bg_color = list_frame.get_bg_color(item_id)
        if str(bg_color) == DISABLE_COLOR:
            list_frame.selection_remove(item_id)


class TestFuncListFrame(BaseListFrame):
    """
    Verification List
    """

    def __init__(self, root=None, select_mode=EXTENDED, size=10):
        values = TestFuncInfo.data_values()

        BaseListFrame.__init__(self, root=root, label=TEST_FUNC_LABEL,
                               columns=TEST_FUNC_HEADERS, values=values,
                               select_mode=select_mode, size=size)


class ModelScriptListFrame(BaseListFrame):
    """
    Model Script List
    """

    def __init__(self, root=None, select_mode=EXTENDED, size=10):

        values = ModelScriptInfo.data_values()
        BaseListFrame.__init__(self, root=root, label=MODEL_SCRIPT_LABEL,
                               columns=MODEL_HEADERS, values=values,
                               select_mode=select_mode, size=size)


class DataScriptListFrame(BaseListFrame):
    """
    Data Script List
    """

    def __init__(self, root=None, select_mode=EXTENDED, size=10):

        values = DataScriptInfo.data_values()
        BaseListFrame.__init__(self, root=root, label=DATA_SCRIPT_LABEL,
                               columns=DATA_HEADERS, values=values,
                               select_mode=select_mode, size=size)
