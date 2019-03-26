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
from tkinter import filedialog, Menu, DISABLED, NORMAL

from conf.configuration import (
    ENV_SETUP_SCRIPT_M_LABEL, DATA_SCRIPT_LABEL, MODEL_SCRIPT_LABEL,
    TEST_FUNC_LABEL, SHOW_ENV_SETUP_M_LABEL,
    RUN_TEST_FUNC, RUN, HELP, SHOW_ENV_SETUP_WIZARD_TITLE,
    ENV_SETUP_WIZARD_TITLE, ADD_NEW_SCRIPT_MSG, INFO, ERROR, ALREADY_SCRIPT_MSG,
    DATA_SCRIPT_WIZARD_TITLE, MODEL_SCRIPT_WIZARD_TITLE, TEST_FUNC_WIZARD_TITLE,
    FILE, ENV_SETUP, DATA_SCRIPT, MODEL_SCRIPT, TEST_FUNC,
    ENV_SET_WIZARD_TITLE, STOP_TEST_FUNC, DELETE_SCRIPTS, DELET_MSG)
from src.com.common import show_wizard, save_config
from src.gui.wizard.env_setup_wizard import EnvSetupWizard
from src.info.data_script_info import DataScriptInfo, DataScript
from src.info.env_setup_info import EnvSetupInfo, EnvSetup
from src.info.model_script_info import ModelScriptInfo, ModelScript
from src.info.test_func_info import TestFuncInfo, TestFunc


class MainMenu(Menu):
    def __init__(self, root=None):
        self.__root = root
        Menu.__init__(self, self.__root)

        # File
        self.__menu_file = Menu(self, tearoff=False)

        # regist env
        self.__menu_file.add_command(label=ENV_SETUP_SCRIPT_M_LABEL,
                                     command=self.__regist_env_setup_script)

        self.__menu_file.add_separator()

        # regist script
        self.__menu_file.add_command(label=DATA_SCRIPT_LABEL,
                                     command=self.__regist_data_script)
        self.__menu_file.add_command(label=MODEL_SCRIPT_LABEL,
                                     command=self.__regist_model_script)
        self.__menu_file.add_command(label=TEST_FUNC_LABEL,
                                     command=self.__regist_test_func_script)

        self.__menu_file.add_separator()

        # delete script
        self.__menu_file.add_command(label=DELETE_SCRIPTS,
                                     command=self.__delete_scripts)

        self.__menu_file.add_separator()

        # show env script list
        self.__menu_file.add_command(label=SHOW_ENV_SETUP_M_LABEL,
                                     command=self.__show_env_wizard)

        self.add_cascade(label=FILE, menu=self.__menu_file)

        # Run
        self.__menu_rnu = Menu(self, tearoff=False)

        # Run Verification
        self.__menu_rnu.add_command(label=RUN_TEST_FUNC, command=self.__run)
        # Stop Verification
        self.__menu_rnu.add_command(label=STOP_TEST_FUNC,
                                    command=self.__stop_test_func)
        self.__menu_rnu.entryconfig(1, state=DISABLED)

        self.add_cascade(label=RUN, menu=self.__menu_rnu)

        # self.add_command(label=HELP, command=show_help)

        self.__main_window = self.__root.children['!mainwindow']
        self.__test_select_frame = self.__main_window.test_select_frame

    def __show_env_wizard(self):
        try:
            self.__main_window.is_running = True
            self.__menu_file.entryconfig(0, state=DISABLED)
            self.__menu_file.entryconfig(6, state=DISABLED)

            env_setup_wizard = EnvSetupWizard(master=self.master,
                                              view_mode=True, size=12)
            show_wizard(env_setup_wizard,
                        SHOW_ENV_SETUP_WIZARD_TITLE, modal=False)

        finally:
            self.__menu_file.entryconfig(0, state=NORMAL)
            self.__menu_file.entryconfig(6, state=NORMAL)
            self.__main_window.is_running = False

    def __regist_env_setup_script(self):
        try:
            self.__main_window.is_running = True
            # Show File Select Dialog
            file_path = filedialog.askopenfilename(
                filetypes=[('All Files', '*.*')], title=ENV_SETUP_WIZARD_TITLE)

            if file_path:

                # Check if it is already registed
                env_setup = EnvSetupInfo.get_data_with_path(file_path)
                if env_setup is None:
                    # Add Env Setup
                    env_setup = EnvSetup(path=file_path)
                    EnvSetupInfo.add_data(env_setup)
                    try:
                        save_config()
                        self.__main_window.c_println(
                            ADD_NEW_SCRIPT_MSG % env_setup.abs_path, mode=INFO)

                    except Exception as e:
                        self.__main_window.c_println(os.linesep + str(e), ERROR)

                else:
                    self.__main_window.c_println(
                        ALREADY_SCRIPT_MSG % (ENV_SETUP, env_setup.id),
                        mode=INFO)
        finally:
            self.__main_window.is_running = False

    def __regist_data_script(self):
        try:
            self.__main_window.is_running = True
            # Show File Select Dialog
            file_path = filedialog.askopenfilename(
                filetypes=[('Python File', '*.py')],
                title=DATA_SCRIPT_WIZARD_TITLE)
            if file_path:
                # show env select wizard
                env_setup_wizard = EnvSetupWizard(master=self.master, size=12)
                show_wizard(wizard=env_setup_wizard, title=ENV_SET_WIZARD_TITLE)

                if env_setup_wizard.env_setup_id < 0:
                    return

                env_id = env_setup_wizard.env_setup_id

                # Check if it is already registed
                data_script = DataScriptInfo.get_data_with_path_eid(
                    path=file_path, env_id=env_id)
                if data_script is None:
                    # Add Data
                    data_script = DataScript(path=file_path, env_id=env_id)
                    DataScriptInfo.add_data(data_script)
                    # Update CONF
                    try:
                        save_config()
                        self.__main_window.c_println(
                            ADD_NEW_SCRIPT_MSG % data_script.abs_path,
                            mode=INFO)

                    except Exception as e:
                        self.__main_window.c_println(os.linesep + str(e), ERROR)
                        return

                else:
                    self.__main_window.c_println(
                        ALREADY_SCRIPT_MSG % (DATA_SCRIPT, data_script.id),
                        mode=INFO)

                # Update MainWindow
                self.__test_select_frame.update_list_items()
        finally:
            self.__main_window.is_running = False

    def __regist_model_script(self):
        try:
            self.__main_window.is_running = True
            # Show File Select Dialog
            file_path = filedialog.askopenfilename(
                filetypes=[('Python File', '*.py')],
                title=MODEL_SCRIPT_WIZARD_TITLE)

            if file_path:
                # show env select wizard
                env_setup_wizard = EnvSetupWizard(master=self.master, size=12)
                show_wizard(wizard=env_setup_wizard, title=ENV_SET_WIZARD_TITLE)

                if env_setup_wizard.env_setup_id < 0:
                    return

                env_id = env_setup_wizard.env_setup_id

                # Check if it is already registed
                model = ModelScriptInfo.get_data_with_path_eid(
                    path=file_path, env_id=env_id)
                if model is None:
                    # Add Model Load
                    model = ModelScript(path=file_path, env_id=env_id)
                    ModelScriptInfo.add_data(model)
                    # Update CONF
                    try:
                        save_config()
                        self.__main_window.c_println(
                            ADD_NEW_SCRIPT_MSG % model.abs_path, mode=INFO)

                    except Exception as e:
                        self.__main_window.c_println(os.linesep + str(e), ERROR)
                        return

                else:
                    self.__main_window.c_println(
                        ALREADY_SCRIPT_MSG % (MODEL_SCRIPT, model.id),
                        mode=INFO)

                # Update MainWindow
                self.__test_select_frame.update_list_items()
        finally:
            self.__main_window.is_running = False

    def __regist_test_func_script(self):
        try:
            self.__main_window.is_running = True
            # Show File Select Dialog
            file_path = filedialog.askopenfilename(
                filetypes=[('Python File', '*.py')],
                title=TEST_FUNC_WIZARD_TITLE)

            if file_path:
                # show env select wizard
                env_setup_wizard = EnvSetupWizard(master=self.master, size=12)
                show_wizard(wizard=env_setup_wizard, title=ENV_SET_WIZARD_TITLE)

                if env_setup_wizard.env_setup_id < 0:
                    return

                env_id = env_setup_wizard.env_setup_id

                # Check if it is already registed
                test_func = TestFuncInfo.get_data_with_path(path=file_path)
                if test_func is None:
                    # Add Test Function
                    test_func = TestFunc(path=file_path, env_id=env_id)
                    TestFuncInfo.add_data(test_func)
                    # Update CONF
                    try:
                        save_config()
                        self.__main_window.c_println(
                            ADD_NEW_SCRIPT_MSG % test_func.abs_path, mode=INFO)

                    except Exception as e:
                        self.__main_window.c_println(os.linesep + str(e), ERROR)
                        return

                else:
                    self.__main_window.c_println(
                        ALREADY_SCRIPT_MSG % (TEST_FUNC, test_func.id),
                        mode=INFO)

                # Update MainWindow
                self.__test_select_frame.update_list_items()
        finally:
            self.__main_window.is_running = False

    def __delete_scripts(self):
        # Delete Scripts
        del_dict = self.__test_select_frame.delete_selected()

        # Save Config
        save_config()

        # Show Deleted Message
        if len(del_dict[TEST_FUNC]) > 0:
            self.__del_print(del_dict[TEST_FUNC])

        if len(del_dict[DATA_SCRIPT]) > 0:
            self.__del_print(del_dict[DATA_SCRIPT])

        if len(del_dict[MODEL_SCRIPT]) > 0:
            self.__del_print(del_dict[MODEL_SCRIPT])

    def __del_print(self, del_list):
        for v in del_list:
            self.__main_window.c_println(DELET_MSG % v, mode=INFO)

    def __run(self):
        try:
            self.__main_window.is_running = True
            self.__menu_rnu.entryconfig(0, state=DISABLED)
            self.__menu_rnu.entryconfig(1, state=NORMAL)

            self.__main_window.test_func_run()

        finally:
            self.__menu_rnu.entryconfig(0, state=NORMAL)
            self.__menu_rnu.entryconfig(1, state=DISABLED)
            self.__main_window.is_running = False

    def __stop_test_func(self):
        self.__main_window.is_running = False
        self.__main_window.stop_test_func()
