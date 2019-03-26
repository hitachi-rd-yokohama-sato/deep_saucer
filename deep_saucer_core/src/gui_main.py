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
import sys
from tkinter import Tk

from conf.configuration import (
    SP_SCREEN_WIDTH, SP_SCREEN_HEIGHT, LINUX_OS,
    MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
from src.gui.splash_screen import SplashScreen
from src.gui.main_window import MainWindow


def show_gui():
    sp_screen = Tk()

    splash_screen = SplashScreen(root=sp_screen, width=SP_SCREEN_WIDTH,
                                 height=SP_SCREEN_HEIGHT)

    splash_screen.update()
    if not splash_screen.startup():
        sys.exit(False)

    m_window = Tk()
    # if os.name == LINUX_OS:
    #     m_window.attributes('-zoomed', '1')

    MainWindow(m_window, width=MAIN_WINDOW_WIDTH, height=MAIN_WINDOW_HEIGHT,
               size=12)

    m_window.focus_force()
    m_window.mainloop()
    # __main_window.mainloop()


if __name__ == '__main__':
    show_gui()
