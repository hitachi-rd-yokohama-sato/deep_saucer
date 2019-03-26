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
import json
import pathlib
from tkinter import (
    Toplevel, Frame, Label, Button, NSEW, TOP, RIGHT, X,
    NE, YES, NW, Entry, StringVar, filedialog, END, DISABLED, BOTH, ACTIVE,
    NORMAL)
from tkinter.scrolledtext import ScrolledText

from conf.configuration import (
    SELECT_CONF_LABEL, NEXT, CANCEL, RUN, FONT_NAME,
    SELECT, PREV, SELECT_USE_CONF_LABEL, VIEW, UTF8, WM_DELETE_WINDOW,
    KEY_RELEASE_EVENT)
from src.com.common import get_geometry, show_wizard


class ConfSelectWizard(Toplevel):

    def __init__(self, master=None, conf_path='',
                 width=0.35, height=0.1, use_factor=True):

        Toplevel.__init__(self, master=master)

        w, h, x, y = get_geometry(self, width, height, use_factor)

        self.geometry('%dx%d+%d+%d' % (w, h, x, y))

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.minsize(int(w), int(h))
        self.maxsize(int(w), int(h))

        # Top Frame
        self.__start_frame = Frame(self)
        # Description og the configuration file
        for msg in SELECT_CONF_LABEL.splitlines(False):
            label = Label(self.__start_frame, text=msg)
            label.config(font=(FONT_NAME, int(w/50)))
            # Deploy
            label.pack(anchor=NW, side=TOP, padx=5)

        # Buttons
        next_btn = Button(self.__start_frame, text=NEXT, width=6,
                          command=self.__next_frame)

        top_run_btn = Button(self.__start_frame, text=RUN, width=6,
                             command=self.__run)

        top_cancel_btn = Button(self.__start_frame, text=CANCEL, width=6,
                                command=self.__cancel)

        # Deploy
        top_cancel_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)
        top_run_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)
        next_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)

        # Selection Frame
        self.__conf_select_frame = Frame(self)
        # Description
        select_title_label = Label(self.__conf_select_frame,
                                   text=SELECT_USE_CONF_LABEL)
        select_title_label.config(font=(FONT_NAME, int(w / 50)))
        select_title_label.pack(anchor=NW, side=TOP, padx=5, pady=5)

        # Conf entry
        entry_frame = Frame(self.__conf_select_frame)
        self.__conf_path_var = StringVar()
        self.__conf_path_var.set(conf_path)
        self.__save_path = conf_path
        conf_entry = Entry(entry_frame, textvariable=self.__conf_path_var)
        conf_entry.config(font=(FONT_NAME, int(w/50)))
        conf_entry.bind(KEY_RELEASE_EVENT, self.__on_change)

        refer_btn = Button(entry_frame, text=SELECT, width=6,
                           command=self.__select_conf)

        # Deploy
        refer_btn.pack(side=RIGHT, padx=5, pady=5)
        conf_entry.pack(side=RIGHT, fill=X, expand=YES, padx=5, pady=5)

        entry_frame.pack(side=TOP, fill=X)

        # Buttons
        prev_btn = Button(self.__conf_select_frame, text=PREV, width=6,
                          command=self.__prev_frame)
        select_cancel_btn = Button(self.__conf_select_frame, text=CANCEL,
                                   width=6, command=self.__cancel)
        self.__select_run_btn = Button(self.__conf_select_frame, text=RUN,
                                       width=6, command=self.__run)
        self.__view_btn = Button(self.__conf_select_frame, text=VIEW, width=6,
                                 command=self.__view_conf)

        # Deploy
        select_cancel_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)
        self.__select_run_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)
        self.__view_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)
        if not conf_path:
            prev_btn.pack(anchor=NE, side=RIGHT, padx=5, pady=3)

        # Deploy
        frame = Frame(self.__conf_select_frame)
        self.__start_frame.grid(row=0, column=0, sticky=NSEW)
        frame.pack()
        self.__conf_select_frame.grid(row=0, column=0, sticky=NSEW)

        self.__is_run = False
        self.__viewing = False

        # If it has already been selected, a selection frame is displayed
        self.__check_conf_path()
        if not conf_path:
            self.__start_frame.tkraise()
        else:
            self.__conf_select_frame.tkraise()

        self.protocol(WM_DELETE_WINDOW, self.__close)

    def __close(self):
        if not self.__viewing:
            self.destroy()

    def __on_change(self, event):
        event.widget.configure(state=DISABLED)
        try:
            self.__check_conf_path()
        finally:
            event.widget.configure(state=NORMAL)

    def __check_conf_path(self):
        # Switching the state of the button
        if (pathlib.Path(self.conf_path).is_file() and
                pathlib.Path(self.conf_path).exists()):

            self.__view_btn.configure(state=ACTIVE)
            self.__select_run_btn.configure(state=ACTIVE)

        elif not self.conf_path:
            self.__view_btn.configure(state=DISABLED)
            self.__select_run_btn.configure(state=ACTIVE)

        else:
            self.__view_btn.configure(state=DISABLED)
            self.__select_run_btn.configure(state=DISABLED)

    @property
    def is_run(self):
        return self.__is_run

    @is_run.deleter
    def is_run(self):
        del self.__is_run

    @property
    def conf_path(self):
        return self.__conf_path_var.get()

    def __next_frame(self):
        if self.__save_path:
            self.__conf_path_var.set(self.__save_path)
        self.__conf_select_frame.tkraise()

    def __prev_frame(self):
        if self.conf_path:
            self.__save_path = self.conf_path

        self.__conf_path_var.set('')
        self.__start_frame.tkraise()

    def __run(self):
        self.__is_run = True
        self.destroy()

    def __cancel(self):
        self.__is_run = False
        self.destroy()

    def __select_conf(self):

        # Show conf file selection dialog
        if self.conf_path:
            init_dir = pathlib.Path(self.conf_path).parent.resolve()
            init_file = pathlib.Path(self.conf_path).name
            file_path = filedialog.askopenfilename(
                parent=self,
                filetypes=[('JSON File', '*.json')],
                title=SELECT_USE_CONF_LABEL,
                initialdir=init_dir,
                initialfile=init_file)
        else:
            file_path = filedialog.askopenfilename(
                parent=self,
                filetypes=[('JSON File', '*.json')],
                title=SELECT_USE_CONF_LABEL)

        if file_path:
            self.__conf_path_var.set(file_path)
            self.__check_conf_path()

    def __view_conf(self):
        try:
            with open(self.conf_path, encoding=UTF8) as rs:
                j_data = json.load(rs)

        except json.JSONDecodeError:
            return

        msg_wizard = Toplevel(master=self.master)

        w, h, x, y = get_geometry(msg_wizard, 0.5, 0.35)

        msg_wizard.geometry('%dx%d+%d+%d' % (w, h, x, y))
        msg_wizard.minsize(int(w), int(h))

        viewer = ScrolledText(msg_wizard, font=(FONT_NAME, 15), height=1)
        viewer.insert(END, json.dumps(j_data, indent=2))
        viewer.configure(state=DISABLED)
        viewer.pack(fill=BOTH, anchor=NW, expand=YES, padx=5, pady=5)

        self.__viewing = True
        show_wizard(msg_wizard, 'Configuration(%s)' % self.conf_path)
        self.__viewing = False
