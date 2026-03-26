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
import sys

__all__ = ["get_pdx_version", "get_version_dict", "show_versions"]


def _get_package_dir():
    """Get the paddlex package directory, compatible with PyInstaller."""
    # When running in a PyInstaller bundle, sys._MEIPASS points to the
    # temporary folder where the bundled files are extracted
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, "paddlex")
    return os.path.dirname(__file__)


def _get_repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _version_from_metadata():
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version("paddlex")
    except PackageNotFoundError:
        return None


def _version_from_setuptools_scm():
    try:
        from setuptools_scm import get_version
    except ImportError:
        return None
    try:
        return get_version(root=_get_repo_root())
    except (LookupError, ValueError):
        return None


def get_pdx_version():
    """Return the installed or source-tree PaddleX version string.

    Resolution order:
    1. `importlib.metadata` (normal pip / wheel install).
    2. `setuptools_scm.get_version` when running from a git checkout.
    3. `\"0.0.0\"` if nothing matches.
    """
    for fn in (
        _version_from_metadata,
        _version_from_setuptools_scm,
    ):
        ver = fn()
        if ver:
            return ver
    return "0.0.0"


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
