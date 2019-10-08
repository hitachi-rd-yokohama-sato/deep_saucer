# -*- coding: utf-8 -*-
import sys
from pathlib import Path

_proj_dir = Path(__file__).absolute().parent.parent
_lib_dir = Path(_proj_dir, 'lib')
_examples_dir = Path(_proj_dir, 'examples')

sys.path.append(str(_proj_dir))
sys.path.append(str(_examples_dir))

from lib.xgb_encoding_verification import main

if __name__ == '__main__':
    model_list = _examples_dir.joinpath('model', '100_3_7', 'model_100_3_7')
    conf_path = str(_examples_dir.joinpath('config.json'))

    main(model_list, None, conf_path)
