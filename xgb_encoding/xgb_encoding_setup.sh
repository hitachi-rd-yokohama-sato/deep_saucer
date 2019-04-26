#!/bin/bash
pip install --upgrade pip
conda create -n xgb_encoding python=3.6 -y
source activate xgb_encoding
pip install pyparsing
pip install pandas
pip install scikit-learn
pip install xgboost
pip install z3-solver