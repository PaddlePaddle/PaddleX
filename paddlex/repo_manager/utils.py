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

import contextlib
import importlib.metadata
import os
import platform
import subprocess
import sys

from ..utils import logging
from ..utils.env import get_device_type

PLATFORM = platform.system()


def _check_call(*args, **kwargs):
    return subprocess.check_call(*args, **kwargs)


def _compare_version(version1, version2):
    import re

    def parse_version(version_str):
        version_pattern = re.compile(
            r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<pre_release>.*))?(?:\+(?P<build_metadata>.+))?$"
        )
        match = version_pattern.match(version_str)
        if not match:
            raise ValueError(f"Unexpected version string: {version_str}")
        return (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
            match.group("pre_release"),
        )

    v1_infos = parse_version(version1)
    v2_infos = parse_version(version2)
    for v1_info, v2_info in zip(v1_infos, v2_infos):
        if v1_info is None and v2_info is None:
            continue
        if v1_info is None or (v2_info is not None and v1_info < v2_info):
            return -1
        if v2_info is None or (v1_info is not None and v1_info > v2_info):
            return 1
    return 0


def check_package_installation(package):
    try:
        importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def install_external_deps(repo_name, repo_root):
    """install paddle repository custom dependencies"""
    import paddle

    def get_gcc_version():
        return subprocess.check_output(["gcc", "--version"]).decode("utf-8").split()[2]

    if repo_name == "PaddleDetection":
        if os.path.exists(os.path.join(repo_root, "ppdet", "ext_op")):
            """Install custom op for rotated object detection"""
            if (
                PLATFORM == "Linux"
                and _compare_version(get_gcc_version(), "8.2.0") >= 0
                and "gpu" in get_device_type()
                and (
                    paddle.is_compiled_with_cuda()
                    and not paddle.is_compiled_with_rocm()
                )
            ):
                with switch_working_dir(os.path.join(repo_root, "ppdet", "ext_op")):
                    # TODO: Apply constraints here
                    args = [sys.executable, "setup.py", "install"]
                    _check_call(args)
            else:
                logging.warning(
                    "The custom operators in PaddleDetection for Rotated Object Detection is only supported when using CUDA, GCC>=8.2.0 and Paddle>=2.0.1, "
                    "your environment does not meet these requirements, so we will skip the installation of custom operators under PaddleDetection/ppdet/ext_ops, "
                    "which means you can not train the Rotated Object Detection models."
                )


def clone_repo_using_git(url, branch=None):
    """clone_repo_using_git"""
    args = ["git", "clone", "--depth", "1"]
    if isinstance(url, str):
        url = [url]
    args.extend(url)
    if branch is not None:
        args.extend(["-b", branch])
    return _check_call(args)


def fetch_repo_using_git(branch, url, depth=1):
    """fetch_repo_using_git"""
    args = ["git", "fetch", url, branch, "--depth", str(depth)]
    _check_call(args)


def reset_repo_using_git(pointer, hard=True):
    """reset_repo_using_git"""
    args = ["git", "reset", "--hard", pointer]
    return _check_call(args)


def remove_repo_using_rm(name):
    """remove_repo_using_rm"""
    if os.path.exists(name):
        if PLATFORM == "Windows":
            return _check_call(["rmdir", "/S", "/Q", name], shell=True)
        else:
            return _check_call(["rm", "-rf", name])


@contextlib.contextmanager
def mute():
    """mute"""
    with open(os.devnull, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield


@contextlib.contextmanager
def switch_working_dir(new_wd):
    """switch_working_dir"""
    cwd = os.getcwd()
    os.chdir(new_wd)
    try:
        yield
    finally:
        os.chdir(cwd)
