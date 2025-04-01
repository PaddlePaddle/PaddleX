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
import importlib.util
import inspect
import re
from collections import defaultdict
from functools import lru_cache, wraps

from packaging.requirements import Requirement

from . import logging

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


def get_dep_version(dep):
    try:
        return importlib.metadata.version(dep)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache()
def is_dep_available(dep, /):
    if dep == "paddlepaddle":
        return importlib.util.find_spec("paddle") is not None
    elif dep == "paddle-custom-device":
        return importlib.util.find_spec("paddle_custom_device") is not None
    elif dep == "ultra-infer":
        return importlib.util.find_spec("ultra_infer") is not None
    return get_dep_version(dep) is not None


def require_deps(*deps, obj=None):
    unavailable_deps = [dep for dep in deps if not is_dep_available(dep)]
    if len(unavailable_deps) > 0:
        if obj is not None:
            msg = f"`{obj.__name__}` is not ready for use, because the"
        else:
            msg = "The"
        msg += "following dependencies are not available:\n" + "\n".join(
            unavailable_deps
        )
        raise RuntimeError(msg)


def function_requires_deps(*deps):
    def _deco(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            require_deps(*func._deps_, obj=func)
            return func(*args, **kwargs)

        func._deps_ = set(deps)
        return _wrapper

    return _deco


def class_requires_deps(*deps):
    def _deco(cls):
        @wraps(cls.__init__)
        def _wrapper(self, *args, **kwargs):
            require_deps(*cls._deps_, obj=cls)
            return cls.__init__(self, *args, **kwargs)

        cls._deps_ = set(deps)
        for base_cls in inspect.getmro(cls):
            if hasattr(base_cls, "_deps_"):
                cls._deps_.update(base_cls._deps_)
        cls.__init__ = _wrapper
        return cls

    return _deco


@lru_cache()
def is_extra_available(extra):
    flags = [is_dep_available(dep) for dep in EXTRAS[extra]]
    if all(flags):
        return True
    logging.debug(
        "These dependencies are not available: %s",
        [d for d, f in zip(EXTRAS[extra], flags) if not f],
    )
    return False


def require_extra(extra, *, obj=None):
    if not is_extra_available(extra):
        if obj is not None:
            msg = f"`{obj.__name__}` requires additional dependencies."
        else:
            msg = "Additional dependencies are required."
        msg += f" To install them, run `pip install paddlex[{extra}]==<PADDLEX_VERSION>` if you’re installing `paddlex` from an index, or `pip install -e /path/to/PaddleX[{extra}]` if you’re installing `paddlex` locally."
        raise RuntimeError(msg)


def function_requires_extra(extra):
    def _deco(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            require_extra(extra, obj=func)
            return func(*args, **kwargs)

        return _wrapper

    return _deco


def class_requires_extra(extra):
    def _deco(cls):
        @wraps(cls.__init__)
        def _wrapper(self, *args, **kwargs):
            require_extra(extra, obj=cls)
            return cls.__init__(self, *args, **kwargs)

        cls.__init__ = _wrapper
        return cls

    return _deco


def is_hpip_available():
    return is_dep_available("ultra-infer")


def require_hpip():
    if not is_hpip_available():
        raise RuntimeError(
            "The high-performance inference plugin is not available. Please install it properly."
        )


def is_serving_plugin_available():
    return is_extra_available("serving")


def require_serving_plugin():
    if not is_serving_plugin_available():
        raise RuntimeError(
            "The serving plugin is not available. Please install it properly."
        )


def is_paddle2onnx_plugin_available():
    return is_extra_available("paddle2onnx")


def require_paddle2onnx_plugin():
    if not is_paddle2onnx_plugin_available():
        raise RuntimeError(
            "The Paddle2ONNX plugin is not available. Please install it properly."
        )
