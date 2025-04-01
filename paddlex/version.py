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


import os

__all__ = ["get_pdx_version", "get_version_dict", "show_versions"]


def get_pdx_version():
    """get_pdx_version"""
    with open(
        os.path.join(os.path.dirname(__file__), ".version"), "r", encoding="ascii"
    ) as fv:
        ver = fv.read().rstrip()
    return ver


def get_version_dict():
    """get_version_dict"""
    import paddle

    from . import repo_manager

    ver_dict = dict()
    ver_dict["pdx"] = get_pdx_version()
    ver_dict["paddle"] = paddle.__version__
    ver_dict["devkits"] = repo_manager.get_versions()
    return ver_dict


def show_versions():
    """show_versions"""
    ver_dict = get_version_dict()
    pdx_ver = f"PDX version: {ver_dict['pdx']}\n"
    paddle_ver = f"PaddlePaddle version: {ver_dict['paddle']}\n"
    repo_vers = []
    for repo_name, vers in ver_dict["devkits"].items():
        sta_ver = vers[0]
        commit = vers[1]
        repo_vers.append(f"{repo_name}:\nversion: {sta_ver}\ncommit id: {commit}\n")
    all_vers = [pdx_ver, paddle_ver, *repo_vers]
    ver_str = "\n".join(all_vers)
    print(ver_str)
