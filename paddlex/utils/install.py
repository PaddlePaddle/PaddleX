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
import subprocess
import sys
import tempfile


def install_packages_from_requirements_file(
    requirements_file_path, pip_install_opts=None
):
    # TODO: Constraints can be applied here to ensure a safe installation.
    # For example, it is best to prevent installing a different version of a
    # distribution for an already loaded package, as that could lead to
    # problems.
    return subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *(pip_install_opts or []),
            "-r",
            requirements_file_path,
        ]
    )


def install_packages(requirements, pip_install_opts=None):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for req in requirements:
            f.write(req + "\n")
        reqs_file_path = f.name
    try:
        return install_packages_from_requirements_file(
            reqs_file_path, pip_install_opts=pip_install_opts
        )
    finally:
        os.unlink(reqs_file_path)


def uninstall_packages(pkgs, pip_uninstall_opts=None):
    return subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "uninstall",
            "-y",
            *(pip_uninstall_opts or []),
            *pkgs,
        ]
    )
