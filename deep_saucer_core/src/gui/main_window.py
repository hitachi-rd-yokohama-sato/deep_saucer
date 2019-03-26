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
import datetime
import os
from tkinter.font import BOLD

from tkinter.scrolledtext import ScrolledText
from tkinter import (
    ttk, Frame, BOTH, NW, YES, HORIZONTAL, SE, TOP, NORMAL, END, DISABLED)

from conf.configuration import (
    DETERMINATE, INFO, WARN, ERROR, BLACK, GOLDENROD, RED,
    INDETERMINATE, LINUX_OS, WM_DELETE_WINDOW,
    VALUE, SELECT_USE_CONF_LABEL,
    NOT_SELECTED_ANACONDA_MSG, CREATE_ENV_MSG,
    CANCEL_MSG, ANACONDA_DIR, TOOL_NAME, DATA_SCRIPT_LABEL, MODEL_SCRIPT_LABEL,
    TEST_FUNC_LABEL, CONSOLE_FONT_NAME)

from src.com.common import (
    cmd_exec_run_posix, show_wizard, get_geometry, create_env,
    create_test_exec_script, save_config, stop_cmd)
from src.gui.main_menu import MainMenu
from src.gui.test_select_frame import TestSelectFrame
from src.gui.wizard.conf_select_wizard import ConfSelectWizard
from src.info.env_setup_info import EnvSetupInfo


class MainWindow(Frame):

    def __init__(self, root=None, width=0.6, height=0.75, size=10,
                 use_factor=True):
        self.__root = root
        Frame.__init__(self, self.__root)
        # MainWindow size
        w, h, x, y = get_geometry(self, width, height, use_factor)

        self.__root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.__root.minsize(int(w), int(h))
        self.lift()

        self.__test_select_frame = TestSelectFrame(self)
        self.__test_select_frame.pack(fill=BOTH, anchor=NW, expand=YES,
                                      padx=5, pady=5)

        # console
        self.__console = ScrolledText(self, font=(CONSOLE_FONT_NAME, size),
                                      height=1)
        self.__console.configure(state=DISABLED)
        self.__console.pack(fill=BOTH, anchor=NW, expand=YES, padx=5, pady=5)

        # Progressbar
        self.__pg_bar = ttk.Progressbar(self, orient=HORIZONTAL,
                                        mode=DETERMINATE)
        self.__pg_bar.configure(maximum=100, value=0)
        self.__pg_bar.pack(anchor=SE, padx=5, pady=5)

        self.__root.title(TOOL_NAME)

        self.pack(side=TOP, fill=BOTH, expand=YES)

        self.__main_menu = MainMenu(self.__root)
        self.__root.configure(menu=self.__main_menu)

        # Console output color
        self.__console.tag_configure(INFO, foreground=BLACK,
                                     font=(CONSOLE_FONT_NAME, size))
        self.__console.tag_configure(WARN, foreground=GOLDENROD,
                                     font=(CONSOLE_FONT_NAME, size))
        self.__console.tag_configure(ERROR, foreground=RED,
                                     font=(CONSOLE_FONT_NAME, size, BOLD))

        self.is_running = False
        self.__root.protocol(WM_DELETE_WINDOW, self.__close)

        self.__user = TOOL_NAME

    def __close(self):
        if not self.is_running:
            self.pg_bar_stop()
            self.__root.destroy()

    def c_println(self, str_val='', mode='normal'):
        self.__console.configure(state=NORMAL)

        split_val = str_val.splitlines()
        for index, value in enumerate(split_val):
            # When it exceeds the maximum line,
            # it deletes it from the first line
            if len(self.__console.get('1.0', END).splitlines()) > 1024 * 4:
                # Delete First Line
                self.__console.delete('1.0', '1.end')
                # Delete Line Feed
                self.__console.delete('1.0')

            # Insert New Line
            if index is 0:
                if mode == 'normal':
                    self.__console.insert(
                        END, '%s : [%s]' % (datetime.datetime.now(),
                                            self.__user), mode)

                else:
                    self.__console.insert(
                        END,
                        '%s : [%s] [%s] ' % (datetime.datetime.now(),
                                             self.__user, mode.upper()), mode)

            self.__console.insert(END, '%s%s' % (value, os.linesep), mode)

            self.__console.see(END)
            self.update()

        self.__console.configure(state=DISABLED)

    @property
    def test_select_frame(self):
        return self.__test_select_frame

    @test_select_frame.deleter
    def test_select_frame(self):
        del self.__test_select_frame

    @property
    def console(self):
        return self.__console

    @console.deleter
    def console(self):
        del self.__console

    @property
    def pg_bar(self):
        return self.__pg_bar

    @pg_bar.deleter
    def pg_bar(self):
        del self.__pg_bar

    def pg_bar_start(self):
        self.__pg_bar.configure(mode=INDETERMINATE)
        self.__pg_bar.start(20)

    def pg_bar_stop(self):
        self.__pg_bar.stop()
        self.__pg_bar.configure(mode=DETERMINATE)
        self.__pg_bar[VALUE] = 0

    def test_func_run(self):
        # Check anaconda
        if ANACONDA_DIR is None:
            self.c_println(NOT_SELECTED_ANACONDA_MSG, ERROR)
            return

        select_test_func_list = self.__test_select_frame.get_select_test_func()
        select_model_list = self.__test_select_frame.get_select_model()
        select_data_list = self.__test_select_frame.get_select_data()

        if len(select_test_func_list) > 0 and len(select_model_list) > 0:
            # Treeview selectmode is BROWSE
            test_func = select_test_func_list[0]
            self.c_println('%s : %s' % (TEST_FUNC_LABEL, test_func.abs_path),
                           mode=INFO)

            if select_data_list:
                data_script = select_data_list[0]
                self.c_println('%s : %s' % (DATA_SCRIPT_LABEL,
                                            data_script.abs_path), mode=INFO)
            else:
                data_script = None

            model_script = select_model_list[0]
            self.c_println('%s : %s' % (MODEL_SCRIPT_LABEL,
                                        model_script.abs_path), mode=INFO)

            # select confing
            conf_select_wizard = ConfSelectWizard(
                master=self.master, conf_path=test_func.conf_path)

            show_wizard(title=SELECT_USE_CONF_LABEL, wizard=conf_select_wizard)
            if not conf_select_wizard.is_run:
                self.c_println(CANCEL_MSG, mode=INFO)
                return

            test_func.conf_path = conf_select_wizard.conf_path

            # Save config
            save_config()

            if test_func.conf_path:
                self.c_println('Configuration : %s' % test_func.conf_path,
                               mode=INFO)
            else:
                self.c_println('Not used configuration file', mode=INFO)

            # Create Execute Environment
            env_setup = EnvSetupInfo.get_data(test_func.env_id)

            env_python_dir = self.__create_env(script_path=env_setup.abs_path)

            if not env_python_dir:
                self.c_println(CANCEL_MSG, mode=INFO)
                return

            self.c_println('Execute Env : %s' % env_python_dir,
                           mode=INFO)

            # template python file replace keyword
            # Created file is deleted after execution
            exec_path = create_test_exec_script(
                test_func=test_func,
                data=data_script,
                model=model_script,
                print_func=self.c_println)

            if exec_path:
                # Run Test Execute Function
                try:
                    self.__call_test_func(
                        test_func=test_func,
                        env_python_dir=env_python_dir,
                        test_func_script_path=exec_path)

                finally:
                    os.remove(exec_path)

    def __create_env(self, script_path):
        # Format of Setup script(sh or bat)
        # Search env name after execute setup script
        # (search 'conda create or python venv')
        # return env directory
        # (AnacondaDir/envs/env_name/bin or AnacondaDir/envs/env_name)
        self.c_println(CREATE_ENV_MSG, mode=INFO)
        self.pg_bar_start()

        # Search Env Name
        try:
            ret_dir = create_env(script_path, self)

        except Exception as e:
            self.c_println(os.linesep + str(e), ERROR)
            return None

        finally:
            self.pg_bar_stop()

        return ret_dir

    def __call_test_func(self, test_func, env_python_dir,
                         test_func_script_path):

        self.c_println('Test Function Start ( %s )' % test_func.abs_path,
                       mode=INFO)
        self.pg_bar_start()

        # Execute environment Python and Script
        cmd = [os.path.join(env_python_dir, 'python'),
               '-u',
               test_func_script_path]

        ret = False
        try:
            # Execute Script
            self.__user = 'Test ID %d' % test_func.id
            if os.name == LINUX_OS:
                ret = cmd_exec_run_posix(command=cmd, main_window=self,
                                         cwd=test_func.dir_path)
            else:
                ret = False

        except Exception as e:
            self.c_println(os.linesep + str(e), ERROR)

        finally:
            self.__user = TOOL_NAME
            self.pg_bar_stop()

        if ret:
            self.c_println('Test Function Complete', mode=INFO)
        else:
            self.c_println('Test Function Failed', mode=INFO)

        return ret

    def stop_test_func(self):
        if stop_cmd():
            self.c_println('Stop!!', WARN)
