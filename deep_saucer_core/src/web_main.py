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
import re
from subprocess import Popen, PIPE

from flask import Flask, request, abort
from pprint import pprint

from conf.configuration import ANACONDA_DIR, INFO, ERROR, DATA_CONF_PATH
from src.com.common import (
    create_env, create_test_exec_script, print_cli)
from src.info.data_script_info import DataScript
from src.info.env_setup_info import EnvSetup
from src.info.model_script_info import ModelScript
from src.info.test_func_info import TestFuncInfo

LOC_ENV_SETUP = 'loc_envsetup'
LOC_MODEL_SCRIPT = 'loc_modelscript'
LOC_DATA_SCRIPT = 'loc_datascript'
LOC_CONF_FILE = 'loc_config'

app = Flask(__name__)


@app.route('/<env>/<test>', methods=['GET', 'POST'])
def run_test_func(env, test):
    """

    :param env: env+id
    :param test: test function script (python) file name
    :return:
    """
    print('----- REQUEST INFO -----')
    print(request.headers)
    pprint(request.view_args)

    j_data = None
    # HACK Confirm in which format the request will be sent.
    if request.is_json and request.data:
        # When the json data is specified
        try:
            j_data = request.json
        except Exception as e:
            abort(e)

    # Other request
    # elif request.form and not request.args:
    #     # When the key-value is specified
    #     j_data = request.form
    #
    # elif request.args and not request.form:
    #     # When a query is specified
    #     j_data = request.args
    #
    # elif request.values:
    #     # Other
    #     j_data = request.values

    else:
        print_cli('Request Pattern not specified', mode=ERROR)
        abort(400)
        pass

    if not j_data:
        print_cli('Argument Error', mode=ERROR)
        abort(400)

    test_func, model, data, env_setup = get_info(env, test, j_data)

    # Create Environment
    env_python_dir = ''
    try:
        env_python_dir = create_env(env_setup.abs_path)

    except Exception as e:
        print(e)
        abort(500)

    if not env_python_dir:
        abort(400)

    # Create execute script
    print_cli('Create execute script', mode=INFO)
    exec_path = create_test_exec_script(test_func, data, model)

    # Run Test Function
    ret_val = run(exec_path, env_python_dir)

    return ret_val


def get_info(env, test, j_data):
    # Get script paths
    loc_env_setup, loc_model_script, loc_data_script, loc_conf_file = get_data(
        j_data)

    # Read configuration
    # ret = read_conf_files()
    ret = TestFuncInfo.read_conf(DATA_CONF_PATH)
    if not ret or not ANACONDA_DIR:
        abort(400)

    # Get information
    # env_id = int(re.sub(r'\D', '', env))
    test_func = TestFuncInfo.get_data_with_name(test)
    env_setup = EnvSetup(path=loc_env_setup, identifier=0)
    model = ModelScript(path=loc_model_script, env_id=env_setup.id,
                        identifier=0)
    # model = ModelScriptInfo.get_data_with_path(loc_model_script)
    if not test_func or not pathlib.Path(model.abs_path).exists():
        print_cli('Incorrect designation of TestFunction or ModelScript',
                  mode=ERROR)
        abort(400)

    if loc_data_script:
        data = DataScript(path=loc_data_script, env_id=env_setup.id,
                          identifier=0)
        # data = DataScriptInfo.get_data_with_path(loc_data_script)

    else:
        data = None

    # configuration file
    test_func.conf_path = loc_conf_file

    return test_func, model, data, env_setup


def run(exec_path, env_python_dir):
    ret_val = ''
    if exec_path:
        # Run
        cmd = list()
        cmd.append(os.path.join(env_python_dir, 'python'))
        cmd.append(exec_path)

        print_cli('Execute Test Function', mode=INFO)
        try:
            proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True)

            std_out, std_err = proc.communicate()
            for val in std_out.splitlines():
                print(val)

            ret_val += std_out

            std_err = re.sub(r'Using\s+[^\s]+\s+backend\.' + os.linesep,
                             '', std_err)
            for val in std_err.splitlines():
                print(val)

            ret_val += std_err

        finally:
            os.remove(exec_path)

    return ret_val


def get_data(data):

    return (data.get(LOC_ENV_SETUP, ''),
            data.get(LOC_MODEL_SCRIPT, ''),
            data.get(LOC_DATA_SCRIPT, ''),
            data.get(LOC_CONF_FILE, ''))


if __name__ == '__main__':
    app.run(debug=False, host='localhost', port=8080)
    # app.run(debug=False, host='0.0.0.0', port=8080)
