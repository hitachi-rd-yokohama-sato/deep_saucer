# -*- coding: utf-8 -*-
from pathlib import Path

_model_list = [
    "./model/100_3_7/model_100_3_7"
]
_parent = Path(__file__).absolute().parent


def model_load(downloaded_data):
    resolve_path_list = []

    for m in _model_list:
        if not Path(m).is_absolute():
            resolve_path_list.append(_parent.joinpath(m).resolve())
        else:
            resolve_path_list.append(m)

    return resolve_path_list
