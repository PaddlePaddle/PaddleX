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


import logging
import os

import chardet
import ruamel.yaml
import yaml
from filelock import FileLock

try:
    import ujson as json
except:
    logging.warning("failed to import ujson, using json instead")
    import json

from contextlib import contextmanager


@contextmanager
def custom_open(file_path, mode):
    """
    自定义打开文件函数

    Args:
        file_path (str): 文件路径
        mode (str): 文件打开模式，'r'，'w' 或 'a'

    Returns:
        Any: 返回文件对象

    Raises:
        FileNotFoundError: 当文件不存在时，raise FileNotFoundError
        ValueError: 当 mode 参数不是 'r'， 'w' 和 'a' 时，raise ValueError
    """
    if mode == "r":
        if not os.path.exists(file_path):
            raise FileNotFoundError("file {} not found".format(file_path))
        file = open(file_path, "r", encoding="utf-8")
        try:
            file.read()
            file.seek(0)
            yield file
        except UnicodeDecodeError:
            file = open(file_path, "r", encoding="gbk")
            try:
                file.read()
                file.seek(0)
                yield file
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    encoding = chardet.detect(f.read())["encoding"]
                file = open(file_path, "r", encoding=encoding)
                yield file
        finally:
            file.close()
    elif mode == "w":
        file = open(file_path, "w", encoding="utf-8")
        yield file
        file.close()
    elif mode == "a":
        encoding = "utf-8"
        if os.path.exists(file_path):
            file = open(file_path, "r", encoding=encoding)
            try:
                file.read()
                file.seek(0)
            except UnicodeDecodeError:
                encoding = "gbk"
                file = open(file_path, "r", encoding=encoding)
                try:
                    file.read()
                    file.seek(0)
                except UnicodeDecodeError:
                    with open(file_path, "rb") as f:
                        encoding = chardet.detect(f.read())["encoding"]
            finally:
                file.close()

        file = open(file_path, "a", encoding=encoding)
        yield file
        file.close()
    else:
        raise ValueError("mode must be 'r', 'w' or 'a', but got {}".format(mode))


# --------------- yaml ---------------


def read_yaml_file(yaml_path: str, to_dict=True):
    """read from yaml file"""
    try:
        with open(yaml_path, "r", encoding="utf-8") as file:
            yaml_content = yaml.full_load(file)
    except UnicodeDecodeError:
        with open(yaml_path, "r", encoding="gbk") as file:
            yaml_content = yaml.full_load(file)
    yaml_content = dict(yaml_content) if to_dict else yaml_content
    return yaml_content


def write_config_file(yaml_dict: dict, yaml_path: str):
    """write to config yaml file"""
    yaml = ruamel.yaml.YAML()
    lock = FileLock(yaml_path + ".lock")
    with lock:
        with open(yaml_path, "w", encoding="utf-8") as file:
            # yaml.safe_dump(yaml_dict, file, sort_keys=False)
            yaml.dump(yaml_dict, file)


def update_yaml_file_with_dict(yaml_path, key_values: dict):
    """update yaml file with key_values
    key_values is a dict
    """
    yaml_dict = read_yaml_file(yaml_path)
    yaml_dict.update(key_values)
    write_config_file(yaml_dict, yaml_path)


def get_yaml_keys(yaml_path):
    """get all keys of yaml file"""
    yaml_dict = read_yaml_file(yaml_path)
    return yaml_dict.keys()


# --------------- markdown ---------------


def generate_markdown_from_dict(metrics):
    """generate_markdown_from_dict"""
    mk = ""
    keys = metrics.keys()
    mk += "| ".join(keys())
    mk += os.linesep
    mk += "|".join([" :----: "])


# ------------------- jsonl ---------------------


def read_jsonl_file(jsonl_path: str):
    """read from jsonl file"""
    with custom_open(jsonl_path, "r") as file:
        jsonl_content = [json.loads(line) for line in file]
    return jsonl_content


def write_json_file(content, jsonl_path: str, ensure_ascii=False, **kwargs):
    """write to json file"""
    with custom_open(jsonl_path, "w") as file:
        json.dump(content, file, ensure_ascii=ensure_ascii, **kwargs)


# --------------- check webui yaml -----------------


def check_dict_keys(to_checked_dict, standard_dict, escape_list=None):
    """check if all keys of to_checked_dict is the same as standard_dict, and the value is the same type
    Args:
        escape_list: if set, will not check the keys in white_list
    """
    escape_list = [] if escape_list is None else escape_list
    for key in standard_dict.keys():
        if key not in to_checked_dict:
            logging.error(f"key {key} not in yaml file")
            return False
        if not isinstance(standard_dict[key], type(to_checked_dict[key])):
            logging.error(
                f"value type of key {key} is not the same as standard: "
                f"{type(standard_dict[key])}, {type(to_checked_dict[key])}"
            )
            return False

        if (
            isinstance(standard_dict[key], dict)
            and isinstance(to_checked_dict[key], dict)
            and key not in escape_list
        ):
            return check_dict_keys(
                to_checked_dict[key], standard_dict[key], escape_list
            )
    if len(to_checked_dict.keys()) != len(standard_dict.keys()):
        logging.error(f"yaml file has extra keys")
        return False

    return True


def check_dataset_valid(path_list):
    """check if dataset valid in path_list for datset_ui"""
    if path_list is not None and len(path_list) > 0:
        for path in path_list:
            if not os.path.exists(path):
                return False
        return True
    else:
        return False
