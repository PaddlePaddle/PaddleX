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

import importlib.metadata
import re
from collections import defaultdict

from packaging.requirements import Requirement

_EXTRA_PATTERN = re.compile(
    r"(?:;|and)*[ \t]*extra[ \t]*==[ \t]*['\"]([a-z0-9]+(?:-[a-z0-9]+)*)['\"]"
)
_EXTRA_NAMES_TO_EXCLUDE = {"base", "plugins"}


def _get_extra_name_and_remove_extra_marker(dep_spec):
    # XXX: Not sure if this is correct
    m = _EXTRA_PATTERN.search(dep_spec)
    if m:
        return m.group(1), dep_spec[: m.start()] + dep_spec[m.end() :]
    else:
        return None, dep_spec


def get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_extras():
    metadata = importlib.metadata.metadata("paddlex")
    extras = {}
    # XXX: The `metadata.get_all` used here is not well documented.
    for name in metadata.get_all("Provides-Extra", []):
        if name not in _EXTRA_NAMES_TO_EXCLUDE:
            extras[name] = defaultdict(list)
    for dep_spec in importlib.metadata.requires("paddlex"):
        extra_name, dep_spec = _get_extra_name_and_remove_extra_marker(dep_spec)
        if extra_name is not None and extra_name not in _EXTRA_NAMES_TO_EXCLUDE:
            dep_spec = dep_spec.rstrip()
            req = Requirement(dep_spec)
            assert extra_name in extras, extra_name
            extras[extra_name][req.name].append(dep_spec)
    return extras


EXTRAS = get_extras()
