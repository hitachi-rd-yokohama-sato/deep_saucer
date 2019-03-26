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
    Toplevel, Button, BOTH, NW, YES, TOP, NE, RIGHT, BROWSE, ACTIVE,
    DISABLED)

from conf.configuration import (
    ENV_SETUP, ENV_SETUP_HEADERS, TREE_VIEW_SELECT_EVENT, OK, CANCEL)
from src.com.common import get_geometry
from src.gui.list_frame import BaseListFrame
from src.info.env_setup_info import EnvSetupInfo


class EnvSetupWizard(Toplevel):

    def __init__(self, master=None, view_mode=False,
                 width=0.2, height=0.45, size=10, use_factor=True):

        Toplevel.__init__(self, master=master)

        w, h, x, y = get_geometry(self, width, height, use_factor)

        self.geometry('%dx%d+%d+%d' % (w, h, x, y))
        # self.minsize(int(w), int(h))
        # self.maxsize(int(w), int(h))

        # EnvSetup List
        self.__env_setup_frame = EnvListFrame(self, size=size)

        # Pack widget
        self.__env_setup_frame.pack(fill=BOTH, anchor=NW, expand=YES, side=TOP)

        if not view_mode:
            # Select Check event
            self.__env_setup_frame.bind_treeview(TREE_VIEW_SELECT_EVENT,
                                                 self.__check_select)
            # Button
            self.__ok_btn = Button(self, text=OK, width=6,
                                   command=self.__select_ok)
            self.__ok_btn.configure(state=DISABLED)

            self.__cancel_btn = Button(self, text=CANCEL, width=6,
                                       command=self.__select_cancel)

            self.__cancel_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=5)
            self.__ok_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=5)

        self.__env_setup_id = -1

    def __check_select(self, event):
        if len(event.widget.selection()) > 0:
            self.__ok_btn.configure(state=ACTIVE)

        else:
            self.__ok_btn.configure(state=DISABLED)

    def __select_ok(self):
        # "selectmode" is "BROWSE", so selected only one
        for item_id in self.__env_setup_frame.get_selection():
            self.__env_setup_id = self.__env_setup_frame.get_item_value(
                item_id=item_id, index=0)
        self.destroy()

    def __select_cancel(self):
        self.destroy()

    @property
    def env_setup_id(self):
        return self.__env_setup_id


class EnvListFrame(BaseListFrame):
    def __init__(self, root=None, size=10):
        values = EnvSetupInfo.data_values()

        BaseListFrame.__init__(self, root=root, label=ENV_SETUP,
                               columns=ENV_SETUP_HEADERS, values=values,
                               select_mode=BROWSE, size=size)
