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
import argparse
import textwrap

from conf.configuration import TOOL_NAME
from src.web_main import app
from src.gui_main import show_gui


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        # formatter_class=argparse.RawDescriptionHelpFormatter,
        usage='Start %(prog)s',
        description=textwrap.dedent(
         """
        If you execute without specifying an option, the %(prog)s GUI will starts up.
        If you specify the -r, --rest option, it starts up as a REST service.
        """),
        add_help=True
    )
    arg_parser.add_argument('-r', '--rest', help='Starts up as a REST service',
                            action='store_true', required=False)

    args = arg_parser.parse_args()

    if args.rest:
        # HACK What number PORT should I use?
        # Starts up as a REST service
        app.run(debug=False, host='0.0.0.0', port=8080)
    else:
        # Starts up as a GUI
        show_gui()
