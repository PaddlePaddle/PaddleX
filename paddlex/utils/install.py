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
    requirements_file_path,
    pip_install_opts=None,
    constraints="base",
):
    from .deps import BASE_DEP_SPECS, REQUIRED_DEP_SPECS

    if constraints not in ("base", "required", "none"):
        raise ValueError(f"Invalid constraints setting: {constraints}")

    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        *(pip_install_opts or []),
        "-r",
        requirements_file_path,
    ]

    if constraints == "base":
        dep_specs = BASE_DEP_SPECS
    elif constraints == "required":
        dep_specs = REQUIRED_DEP_SPECS
    else:
        dep_specs = None
    if dep_specs:
        # TODO: Precompute or cache the constraints
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for reqs in dep_specs.values():
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
        args.extend(["-c", constraints_file_path])

    logging.debug("Command: %s", args)

    try:
        return subprocess.check_call(args)
    finally:
        os.unlink(constraints_file_path)


def install_packages(requirements, pip_install_opts=None, constraints="base"):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for req in requirements:
            f.write(req + "\n")
        reqs_file_path = f.name
    try:
        return install_packages_from_requirements_file(
            reqs_file_path,
            pip_install_opts=pip_install_opts,
            constraints=constraints,
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
