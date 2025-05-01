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

from packaging.requirements import Requirement

from . import logging


def install_packages_from_requirements_file(
    requirements_file_path, pip_install_opts=None
):
    from .deps import DEP_SPECS

    # TODO: Precompute or cache the constraints
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for reqs in DEP_SPECS.values():
            for req in reqs:
                req = Requirement(req)
                if req.marker and not req.marker.evaluate():
                    continue
                if req.url:
                    req = f"{req.name}@{req.url}"
                else:
                    req = f"{req.name}{req.specifier}"
                f.write(req + "\n")
        constraints_file_path = f.name

    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-c",
        constraints_file_path,
        *(pip_install_opts or []),
        "-r",
        requirements_file_path,
    ]
    logging.debug("Command: %s", args)

    try:
        return subprocess.check_call(args)
    finally:
        os.unlink(constraints_file_path)


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


def uninstall_packages(packages, pip_uninstall_opts=None):
    args = [
        sys.executable,
        "-m",
        "pip",
        "uninstall",
        "-y",
        *(pip_uninstall_opts or []),
        *packages,
    ]
    logging.debug("Command: %s", args)
    return subprocess.check_call(args)
