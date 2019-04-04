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
import pathlib
from subprocess import Popen, PIPE

# GUI Settings
TOOL_NAME = 'DeepSaucer'
SP_SCREEN_COLOR = 'spring green'
COPY_RIGHT = \
    'Copyright(c) 2019.\n' \
    'School of Electronics and Computer Science, ' \
    'University of Southampton\n' \
    'and Hitachi, Ltd. All Rights Reserved.'

SP_SCREEN_WIDTH = 0.3*1.5
SP_SCREEN_HEIGHT = 0.2*1.5

MAIN_WINDOW_WIDTH = 0.6
MAIN_WINDOW_HEIGHT = 0.75

DISABLE_COLOR = 'gray66'
SELECTED_COLOR = 'light gray'
BLACK = 'black'
GOLDENROD = 'goldenrod'
RED = 'red'
LINE_COLORS = ['alice blue', 'white']

# Check available fonts
# tkinter.Tk()
# tkinter.font.families()
FONT_NAME = 'bitstream charter'
CONSOLE_FONT_NAME = 'bitstream charter'

TREE_VIEW = 'Treeview'
TREE_VIEW_HEAD = 'Treeview.Heading'
HEADINGS = 'headings'
VALUES = 'values'
VALUE = 'value'
BACK_GROUND = 'background'
DETERMINATE = 'determinate'
INDETERMINATE = 'indeterminate'

WM_DELETE_WINDOW = 'WM_DELETE_WINDOW'

ESCAPE_EVENT = '<Escape>'
TREE_VIEW_SELECT_EVENT = '<<TreeviewSelect>>'
KEY_RELEASE_EVENT = '<KeyRelease>'
L_DOUBL_CLICK_EVENT = '<Double-Button-1>'
ENTER_KEY_EVENT = '<Return>'

TEST_FUNC_HEADERS = [' ID ', 'ENV ID', '   Name   ', '   Path   ']
MODEL_HEADERS = [' ID ', 'ENV ID', '   Name   ', '   Path   ']
DATA_HEADERS = [' ID ', 'ENV ID', '   Name   ', '   Path   ']
ENV_SETUP_HEADERS = [' ID ', '   Name   ', '   Path   ']

# Label and Title
ENV_SETUP_LABEL = 'Env Setup Script'
SHOW_ENV_SETUP_M_LABEL = 'Show %s' % ENV_SETUP_LABEL
SELECT_CONF_LABEL = \
    'Configuration file is not spcecified.\n' \
    'To specify the configuration file, select "Next".\n' \
    'To execute the test function, select "Run."'
SELECT_USE_CONF_LABEL = 'Select the configuration file to use'
RUN_TEST_FUNC = 'Run Test Function'
STOP_TEST_FUNC = 'Stop Test Function'
DELETE_SCRIPTS = 'Delete Scripts'
TEST_FUNC_LABEL = 'Verification Script'
DATA_SCRIPT_LABEL = 'Test Dataset Load Script'
MODEL_SCRIPT_LABEL = 'Model Load Script'
COMMENT_LABEL = '* Double click to edit the linked Env Setup Scripts'

ADD_ENV_SETUP_SCRIPT_LABEL = 'Add %s' % ENV_SETUP_LABEL
ADD_TEST_FUNC_LABEL = 'Add %s' % TEST_FUNC_LABEL
ADD_MODEL_SCRIPT_LABEL = 'Add %s' % MODEL_SCRIPT_LABEL
ADD_DATA_SCRIPT_LABEL = 'Add %s' % DATA_SCRIPT_LABEL

SHOW_ENV_SETUP_WIZARD_TITLE = 'Registered %s' % ENV_SETUP_LABEL
ENV_SETUP_WIZARD_TITLE = 'Select %s' % ENV_SETUP_LABEL
DATA_SCRIPT_WIZARD_TITLE = 'Select %s' % DATA_SCRIPT_LABEL
MODEL_SCRIPT_WIZARD_TITLE = 'Select %s' % MODEL_SCRIPT_LABEL
TEST_FUNC_WIZARD_TITLE = 'Select %s' % TEST_FUNC_LABEL
ENV_SET_WIZARD_TITLE = 'Select Execute Env'
ENV_EDIT_WIZARD_TITLE = 'Environment Edit Wizard (%s)'

OK = 'OK'
CANCEL = 'Cancel'
NEXT = 'Next'
PREV = 'Prev'
SELECT = 'Select'
VIEW = 'View'
ADD = 'Add'
DEL = 'Delete'

# Message
ADD_NEW_SCRIPT_MSG = 'Add New Script (%s)'
ALREADY_SCRIPT_MSG = '%s has already been registered (ID:%d)'
NOT_SELECTED_ANACONDA_MSG = 'Anaconda Directory is not selected'
CREATE_ENV_MSG = 'Create Test Execute Enviroment'
ALREADY_CREATED_ENV_MSG = 'Already Created Env Directory (%s)'
CANCEL_MSG = 'Execution was canceled'
DELET_MSG = 'Deleted %s (ID: %d, PATH: %s)'

# KeyWord
ENV_SETUP = 'EnvSetup'
DATA_SCRIPT = 'DataScript'
MODEL_SCRIPT = 'ModelScript'
TEST_FUNC = 'TestFunc'
CONF_PATH = 'ConfPath'

ID = 'ID'
PATH = 'Path'
FILE = 'File'
RUN = 'Run'
HELP = 'Help'
INFO = 'info'
WARN = 'warn'
ERROR = 'error'

LINKED_ENV = 'Linked Env Setup'
NON_LINKED_ENV = 'Non-linked Env Setup'

LINUX_OS = 'posix'

UTF8 = 'utf-8'
SHIFT_JIS = 'shift_jis'

ENVS = 'envs'
BIN = 'bin'

SYS_PATH_APPEND = "sys.path.append(r'%s')"
CALL_PRINT = "print('[INFO] Call %s')"
FLUSH = 'sys.stdout.flush()'
LOAD_CHECK_FORMAT = \
    "    if {0} is None:\n" \
    "        sys.stderr.write('Failed to acquire {0}')\n" \
    "        sys.exit(1)\n"

# Paths
__proc = Popen(['which', 'conda'], stdout=PIPE, stderr=PIPE,
               universal_newlines=True)
CONDA, _ = __proc.communicate()
CONDA = CONDA.strip()
__proc.terminate()
__proc = Popen([CONDA, 'info', '--base'], stdout=PIPE, stderr=PIPE,
               universal_newlines=True)
ANACONDA_DIR, _ = __proc.communicate()
ANACONDA_DIR = ANACONDA_DIR.strip()
__proc.terminate()
_home = pathlib.Path(__file__).parent.parent.resolve()
HOME = str(_home)
_conf_dir = _home.joinpath('conf')
CONF_DIR = str(_conf_dir)
DATA_CONF_PATH = _conf_dir.joinpath('DATA_CONF.yml')
TMP_DIR = str(_home.joinpath('tmp'))
if not os.path.exists(TMP_DIR):
    os.mkdir(TMP_DIR)
TMP_TEST_EXECUTE = 'tmp_test_exec.py'
DOWNLOAD = _home.joinpath('downloaded_data')
if not os.path.exists(DOWNLOAD):
    os.mkdir(DOWNLOAD)
