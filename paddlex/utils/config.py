# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os

import yaml

from . import logging
from .file_interface import custom_open

__all__ = ["get_config"]


class AttrDict(dict):
    """Attr Dict"""

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


def create_attr_dict(yaml_config):
    """create attr dict"""
    from ast import literal_eval

    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""
    with custom_open(cfg_file, "r") as fopen:
        yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
    create_attr_dict(yaml_config)
    return yaml_config


def print_dict(d, delimiter=0):
    """
    Recursively visualize a dict and
    indenting according by the relationship of keys.
    """
    placeholder = "-" * 60
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            logging.info("{}{} : ".format(delimiter * " ", k))
            print_dict(v, delimiter + 4)
        elif isinstance(v, list) and len(v) >= 1 and isinstance(v[0], dict):
            logging.info("{}{} : ".format(delimiter * " ", k))
            for value in v:
                print_dict(value, delimiter + 4)
        else:
            logging.info("{}{} : {}".format(delimiter * " ", k, v))
        if k.isupper():
            logging.info(placeholder)


def print_config(config):
    """
    visualize configs
    Arguments:
        config: configs
    """
    logging.advertise()
    print_dict(config)


def override(dl, ks, v):
    """
    Recursively replace dict of list
    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    def parse_str(s):
        """convert str type value
        to None type if it is "None",
        to bool type if it means True or False,
        to int type if it can be eval().
        """
        if s in ("None"):
            return None
        elif s in ("TRUE", "True", "true", "T", "t"):
            return True
        elif s in ("FALSE", "False", "false", "F", "f"):
            return False
        try:
            return eval(v)
        except Exception:
            return s

    assert isinstance(dl, (list, dict)), "{} should be a list or a dict"
    assert len(ks) > 0, "length of keys should larger than 0"
    if isinstance(dl, list):
        k = parse_str(ks[0])
        if len(ks) == 1:
            assert k < len(dl), "index({}) out of range({})".format(k, dl)
            dl[k] = parse_str(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            # assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                logging.warning(f"A new field ({ks[0]}) detected!")
            dl[ks[0]] = parse_str(v)
        else:
            if ks[0] not in dl.keys():
                dl[ks[0]] = {}
                logging.warning(f"A new Series field ({ks[0]}) detected!")
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config
    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]
    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), "option({}) should be a str".format(opt)
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt)
            )
            pair = opt.split("=")
            assert len(pair) == 2, "there can be only a = in the option"
            key, value = pair
            keys = key.split(".")
            override(config, keys, value)
    return config


def get_config(fname, overrides=None, show=False):
    """
    Read config from file
    """
    assert os.path.exists(fname), "config file({}) is not exist".format(fname)
    config = parse_config(fname)
    override_config(config, overrides)
    if show:
        print_config(config)
    # check_config(config)
    return config


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser("PaddleX script")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/config.yaml",
        help="config file path",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        help="config options to be overridden",
    )
    parser.add_argument(
        "-p",
        "--profiler_options",
        type=str,
        default=None,
        help='The option of profiler, which should be in format "key1=value1;key2=value2;key3=value3".',
    )
    args = parser.parse_args()
    return args
