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
import re

import select
from select import POLLIN

from collections import OrderedDict
from subprocess import Popen, PIPE
from pathlib import Path
from stat import S_IREAD, S_IEXEC
from tkinter import messagebox

import yaml

from conf.configuration import (
    DATA_CONF_PATH, UTF8, LINUX_OS,
    SHIFT_JIS, ENVS, BIN, ALREADY_CREATED_ENV_MSG, TMP_DIR, TMP_TEST_EXECUTE,
    ERROR, INFO, ANACONDA_DIR, TEST_FUNC_LABEL, MODEL_SCRIPT_LABEL,
    DATA_SCRIPT_LABEL, DOWNLOAD, SYS_PATH_APPEND, CALL_PRINT, LOAD_CHECK_FORMAT,
    FLUSH)
from src.info.data_script_info import DataScriptInfo
from src.info.env_setup_info import EnvSetupInfo
from src.info.model_script_info import ModelScriptInfo
from src.info.test_func_info import TestFuncInfo

proc = None
killed = False


# def show_help():
#     # TODO Ubuntu DisplayServer Wayland is NG, Xorg is OK
#     messagebox.showinfo('Help', 'Help!')


def read_config():
    """
    Reading registered script information
    :return:
    """
    # Env Setup Info Read
    if not EnvSetupInfo.read_conf(DATA_CONF_PATH):
        return False
    # Data Script Read
    if not DataScriptInfo.read_conf(DATA_CONF_PATH):
        return False
    # Model Script Read
    if not ModelScriptInfo.read_conf(DATA_CONF_PATH):
        return False
    # Verification Reaad
    if not TestFuncInfo.read_conf(DATA_CONF_PATH):
        return False

    return True


def save_config():
    """
    Saveing registered script information
    :return:
    """
    with open(file=DATA_CONF_PATH, mode='w', encoding=UTF8) as wf:
        # Write EnvSetup
        wf.write(yaml.dump(EnvSetupInfo.data(), default_flow_style=False))
        wf.write(os.linesep)

        # Write DataScript
        wf.write(
            yaml.dump(DataScriptInfo.data(), default_flow_style=False))
        wf.write(os.linesep)

        # Write ModelLoaad
        wf.write(
            yaml.dump(ModelScriptInfo.data(), default_flow_style=False))
        wf.write(os.linesep)

        # Write TestFunc
        wf.write(yaml.dump(TestFuncInfo.data(), default_flow_style=False))
        wf.write(os.linesep)


def __represent_odict(dumper, instance):
    return dumper.represent_mapping('tag:yaml.org,2002:map', instance.items())


def setup_yaml():
    # Use OrderedDict
    yaml.add_representer(OrderedDict, __represent_odict)


def get_select_value(select_frame, info):
    result = []
    for item_id in select_frame.get_selection():
        identifier = select_frame.get_item_value(item_id, 0)
        result.append(info.get_data(identifier))
    return result


def cmd_exec_run_posix(command, main_window, shell=False, cwd=None):
    global killed
    global proc

    killed = False

    try:
        for std_out in __execute_posix(command=command, main_window=main_window,
                                       shell=shell, cwd=cwd):
            main_window.update()
            if type(std_out) is str:
                main_window.c_println(std_out)

            elif type(std_out) is bool:
                return std_out
    finally:
        proc = None


def __execute_posix(command, main_window, shell=False, cwd=None):
    if shell and type(command) is list:
        cmd = ' '.join(command)
    else:
        cmd = command

    global proc
    proc = Popen(cmd, shell=shell, stdout=PIPE, stderr=PIPE,
                 universal_newlines=True, cwd=cwd)

    # Non block readline(Linux)
    poll_obj = select.poll()
    poll_obj.register(proc.stdout, POLLIN)

    try:
        while True:
            # init std_out
            std_out = ''
            main_window.update()

            if poll_obj:
                poll_result = poll_obj.poll(0)
                if poll_result:
                    std_out = proc.stdout.readline()

            # return std_out and show console
            if std_out and not killed:
                yield std_out

                continue

            if not std_out and proc.poll() is not None:
                break

        _, std_err = proc.communicate()

        std_err = re.sub(r'Using\s+[^\s]+\s+backend\.' + os.linesep,
                         '', std_err)
        if std_err:
            main_window.c_println(os.linesep + std_err, 'error')

    finally:
        if proc.returncode is not 0:
            ret = False
        else:
            ret = True

        if proc.poll() is not None:
            proc.kill()

        proc = None

    yield ret


def stop_cmd():
    """
    Forcibily terminate the process
    :return:
    """
    global proc
    if proc:
        global killed
        killed = True
        proc.kill()
        return True
    else:
        return False


def show_wizard(wizard, title, modal=True):
    wizard.lift()
    wizard.title(title)
    wizard.focus_set()
    wizard.transient(wizard.master)

    if modal:
        wizard.grab_set()

    wizard.wait_window()


def get_geometry(frame, width, height, use_factor=True):
    ws = frame.winfo_screenwidth()
    hs = frame.winfo_screenheight()
    w = (use_factor and ws * width) or width
    h = (use_factor and hs * height) or height

    x = (ws / 2) - (w / 2)
    y = (hs / 2) - (h / 2)

    return w, h, x, y


def create_env(script_path, window=None):
    global killed
    global proc

    killed = False
    pattern = r'\s*conda\s+create\s+-(-name|[^n\s]*n[^n\s]*)\s+([^\s]+)'
    if window:
        print_func = window.c_println
    else:
        print_func = print_cli

    # Save Original Permission
    permission = Path(script_path).stat().st_mode

    try:
        if os.name == LINUX_OS:
            encod = UTF8
        else:
            encod = SHIFT_JIS

        # Add Read Permission
        if not permission or S_IREAD > 0:
            Path(script_path).chmod(permission | S_IREAD)

        env_name = ''
        with open(file=script_path, mode='r', encoding=encod) as read_file:
            value = read_file.readline()
            while value:
                m = re.match(pattern, value)
                if m:
                    # Apply first match line
                    env_name = m.groups()[1]
                    break

                value = read_file.readline()

        if not env_name:
            raise Exception('[Error] Not Find Env Name')

        ret_dir = os.path.join(ANACONDA_DIR, ENVS, env_name)
        if os.name == LINUX_OS:
            ret_dir = os.path.join(ret_dir, BIN)

            # Already Env Directory Created
            if os.path.exists(ret_dir):
                print_func(ALREADY_CREATED_ENV_MSG % ret_dir, mode=INFO)

                return ret_dir

        # Add Exec Permission
        if not permission or S_IEXEC > 0:
            Path(script_path).chmod(permission | S_IEXEC)

        # Execute Script
        if window:
            if os.name == LINUX_OS:
                ret_cmd = cmd_exec_run_posix(command=script_path,
                                             main_window=window)
            else:
                ret_cmd = False
        else:
            proc = Popen(script_path, universal_newlines=True)
            proc.wait()

            if proc.returncode is 0:
                ret_cmd = True

            else:
                ret_cmd = False

        if not ret_cmd:
            return None

        else:
            return ret_dir

    except Exception as e:
        raise e

    finally:
        # Set Original Permission
        Path(script_path).chmod(permission)


def print_cli(value, mode='normal'):

    if mode == 'normal':
            print('%s : %s' % (datetime.datetime.now(), value))

    else:
        print('%s : [%s] %s' % (datetime.datetime.now(), mode.upper(), value))


def create_test_exec_script(test_func, data, model, print_func=print_cli):

    now_str = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    exec_path = os.path.join(TMP_DIR, now_str + '_' + TMP_TEST_EXECUTE)
    tmp_path = os.path.join(TMP_DIR, TMP_TEST_EXECUTE)

    try:
        # Create Call Test Execute Script
        with open(file=exec_path, mode='w', encoding=UTF8) as ws,\
                open(file=tmp_path, mode='r', encoding=UTF8) as rs:

            line = rs.readline()
            while line:
                if line.strip() == '# ## PATH_APPEND ## #':
                    # append script directory
                    if data:
                        _writeline(ws, SYS_PATH_APPEND % data.dir_path, 1)
                    _writeline(ws, SYS_PATH_APPEND % model.dir_path, 1)
                    _writeline(ws, SYS_PATH_APPEND % test_func.dir_path, 1)

                elif line.strip() == '# ## IMPORT_SCRIPT ## #':
                    # import script
                    _writeline(ws, "print('[INFO] Import Script')", 1)
                    _writeline(ws, FLUSH, 1)
                    if data:
                        d_name, _ = os.path.splitext(data.name)
                        _writeline(ws, 'import %s as dc' % d_name, 1)

                    m_name, _ = os.path.splitext(model.name)
                    _writeline(ws, 'import %s as ml' % m_name, 1)

                    t_name, _ = os.path.splitext(test_func.name)
                    _writeline(ws, 'import %s as tf' % t_name, 1)

                elif line.strip() == '# ## CALL_DATA_LOADER ## #' and data:
                    # Call Data Script
                    _writeline(ws, CALL_PRINT % DATA_SCRIPT_LABEL, 1)
                    _writeline(ws, FLUSH, 1)
                    _writeline(ws, "data = dc.data_create('%s')" % DOWNLOAD, 1)
                    _writeline(ws, FLUSH, 1)
                    _writeline(ws, '')

                    # Call Data Script
                    ws.write(LOAD_CHECK_FORMAT.format('data'))

                elif line.strip() == '# ## CALL_MODEL_LOADER ## #':
                    # Call Model Script
                    _writeline(ws, CALL_PRINT % MODEL_SCRIPT_LABEL, 1)
                    _writeline(ws, FLUSH, 1)
                    _writeline(ws, "model = ml.model_load('%s')" % DOWNLOAD, 1)
                    _writeline(ws, FLUSH, 1)
                    _writeline(ws, '')

                    # Model Check
                    ws.write(LOAD_CHECK_FORMAT.format('model'))

                elif line.strip() == '# ## CALL_TEST_FUNCTION ## #':
                    # Call Test Function
                    _writeline(ws, CALL_PRINT % TEST_FUNC_LABEL, 1)
                    _writeline(ws, FLUSH, 1)
                    _writeline(ws, 'time.sleep(1)', 1)
                    if data and test_func.conf_path:
                        args_str = 'model, data, "%s"' % test_func.conf_path

                    elif data and not test_func.conf_path:
                        args_str = 'model, data'

                    elif not data and test_func.conf_path:
                        args_str = 'model, None, "%s"' % test_func.conf_path

                    else:
                        args_str = 'model'

                    _writeline(ws, 'result = tf.main(%s)' % args_str, 1)

                else:
                    ws.write(line)

                line = rs.readline()

    except Exception as e:
        print_func(os.linesep + str(e), ERROR)

        if os.path.exists(exec_path):
            os.remove(exec_path)
        return None

    return exec_path


def _writeline(ws, val, tab=0):
    for i in range(tab):
        ws.write('    ')

    ws.write('%s\n' % val)
