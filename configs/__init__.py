"""
Name : __init__.py
Author  : Hanat
Time    : 2019-12-24 14:38
Desc:
"""

import os
import yaml


def init_config():
    basedir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../configs')
    with open(os.path.join(basedir, "config.yml"), 'r') as f:
        conf = yaml.load(f.read(), Loader=yaml.Loader)
    return conf


conf = init_config()
