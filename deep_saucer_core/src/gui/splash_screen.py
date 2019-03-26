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
from tkinter import (
    Frame, CENTER, TOP, BOTH, YES, ttk, HORIZONTAL, BOTTOM, SW, LEFT, Label)
from tkinter.font import BOLD

from conf.configuration import (
    SP_SCREEN_COLOR, COPY_RIGHT, DATA_CONF_PATH, FONT_NAME,
    DETERMINATE, VALUE, TOOL_NAME)

from src.com.common import read_config, setup_yaml, get_geometry


class SplashScreen(Frame):

    def __init__(self, root=None, width=0.3, height=0.2, use_factor=True):
        self.root = root

        Frame.__init__(self, self.root)

        # size
        w, h, x, y = get_geometry(self, width, height, use_factor)

        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.root.overrideredirect(True)
        self.lift()

        self.config(bg=SP_SCREEN_COLOR)

        # Title Label
        title = Label(self, text=TOOL_NAME)
        title.config(bg=SP_SCREEN_COLOR, justify=CENTER,
                     font=(FONT_NAME, int(w / 10), BOLD))
        title.pack(side=TOP, fill=BOTH, expand=YES, pady=int(w / 35))

        # Progressbar
        self.pg_bar = ttk.Progressbar(self, orient=HORIZONTAL,
                                      mode=DETERMINATE, length=w)
        self.pg_bar.pack(side=BOTTOM, anchor=SW, expand=YES)

        # copyright
        copy_right = Label(self, text=COPY_RIGHT)
        copy_right.config(bg=SP_SCREEN_COLOR, justify=LEFT,
                          font=(FONT_NAME, int(w / 50)))
        copy_right.pack(side=BOTTOM, anchor=SW, expand=YES, padx=5)

        self.pack(side=TOP, fill=BOTH, expand=YES)

    def startup(self):
        """
        startup
        :return:
        """
        self.pg_bar.start()

        # setup_yaml
        setup_yaml()
        # read config
        result = True
        if os.path.exists(DATA_CONF_PATH):
            result = read_config()

        self.pg_bar.stop()
        self.pg_bar[VALUE] = 1000 * 2
        self.pg_bar.config(value=self.pg_bar[VALUE])

        self.update()
        from time import sleep
        sleep(1.5)

        # Delete SplashScree
        self._root().destroy()

        return result
