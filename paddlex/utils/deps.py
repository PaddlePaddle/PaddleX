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
from packaging.version import Version

from . import logging

_EXTRA_PATTERN = re.compile(
    r"(?:;|and)*[ \t]*extra[ \t]*==[ \t]*['\"]([a-z0-9]+(?:-[a-z0-9]+)*)['\"]"
)
_COLLECTIVE_EXTRA_NAMES = {"base", "plugins", "all"}


def _get_extra_name_and_remove_extra_marker(dep_spec):
    # XXX: Not sure if this is correct
    m = _EXTRA_PATTERN.search(dep_spec)
    if m:
        return m.group(1), dep_spec[: m.start()] + dep_spec[m.end() :]
    else:
        return None, dep_spec


def _get_extras():
    metadata = importlib.metadata.metadata("paddlex")
    extras = {}
    # XXX: The `metadata.get_all` used here is not well documented.
    for name in metadata.get_all("Provides-Extra", []):
        if name not in _COLLECTIVE_EXTRA_NAMES:
            extras[name] = defaultdict(list)
    for dep_spec in importlib.metadata.requires("paddlex"):
        extra_name, dep_spec = _get_extra_name_and_remove_extra_marker(dep_spec)
        if extra_name is not None and extra_name not in _COLLECTIVE_EXTRA_NAMES:
            dep_spec = dep_spec.rstrip()
            req = Requirement(dep_spec)
            assert extra_name in extras, extra_name
            extras[extra_name][req.name].append(dep_spec)
    return extras


EXTRAS = _get_extras()


def _get_dep_specs():
    dep_specs = defaultdict(list)
    for dep_spec in importlib.metadata.requires("paddlex"):
        extra_name, dep_spec = _get_extra_name_and_remove_extra_marker(dep_spec)
        if extra_name is None or extra_name == "all":
            dep_spec = dep_spec.rstrip()
            req = Requirement(dep_spec)
            dep_specs[req.name].append(dep_spec)
    return dep_specs


DEP_SPECS = _get_dep_specs()


def get_dep_version(dep):
    try:
        return importlib.metadata.version(dep)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache()
def is_dep_available(dep, /, check_version=None):
    # Currently for several special deps we check if the import packages exist.
    if dep in ("paddlepaddle", "paddle-custom-device", "ultra-infer") and check_version:
        raise ValueError(
            "Currently, `check_version` is not allowed to be `True` for `paddlepaddle`, `paddle-custom-device`, and `ultra-infer`."
        )
    if dep == "paddlepaddle":
        return importlib.util.find_spec("paddle") is not None
    elif dep == "paddle-custom-device":
        return importlib.util.find_spec("paddle_custom_device") is not None
    elif dep == "ultra-infer":
        return importlib.util.find_spec("ultra_infer") is not None
    else:
        if dep != "paddle2onnx" and dep not in DEP_SPECS:
            raise ValueError("Unknown dependency")
    if check_version is None:
        if dep == "paddle2onnx":
            check_version = True
        else:
            check_version = False
    version = get_dep_version(dep)
    if version is None:
        return False
    if check_version:
        if dep == "paddle2onnx":
            return Version(version) in Requirement(get_paddle2onnx_spec()).specifier
        for dep_spec in DEP_SPECS[dep]:
            if Version(version) in Requirement(dep_spec).specifier:
                return True
    else:
        return True


def require_deps(*deps, obj_name=None):
    unavailable_deps = [dep for dep in deps if not is_dep_available(dep)]
    if len(unavailable_deps) > 0:
        if obj_name is not None:
            msg = f"`{obj_name}` is not ready for use, because the"
        else:
            msg = "The"
        msg += " following dependencies are not available:\n" + "\n".join(
            unavailable_deps
        )
        raise RuntimeError(msg)


def function_requires_deps(*deps):
    def _deco(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            require_deps(*func._deps_, obj_name=func.__name__)
            return func(*args, **kwargs)

        func._deps_ = set(deps)
        return _wrapper

    return _deco


def class_requires_deps(*deps):
    def _deco(cls):
        @wraps(cls.__init__)
        def _wrapper(self, *args, **kwargs):
            require_deps(*cls._deps_, obj_name=cls.__name__)
            return old_init_func(self, *args, **kwargs)

        cls._deps_ = set(deps)
        for base_cls in inspect.getmro(cls)[1:-1]:
            if hasattr(base_cls, "_deps_"):
                cls._deps_.update(base_cls._deps_)
        if "__init__" in cls.__dict__:
            old_init_func = cls.__init__
        else:

            def _forward(self, *args, **kwargs):
                return super(cls, self).__init__(*args, **kwargs)

            old_init_func = _forward
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


def require_extra(extra, *, obj_name=None):
    if not is_extra_available(extra):
        if obj_name is not None:
            msg = f"`{obj_name}` requires additional dependencies."
        else:
            msg = "Additional dependencies are required."
        msg += f' To install them, run `pip install "paddlex[{extra}]==<PADDLEX_VERSION>"` if you’re installing `paddlex` from an index, or `pip install -e "/path/to/PaddleX[{extra}]"` if you’re installing `paddlex` locally.'
        raise RuntimeError(msg)


def pipeline_requires_extra(extra):
    def _deco(pipeline_cls):
        @wraps(pipeline_cls.__init__)
        def _wrapper(self, *args, **kwargs):
            require_extra(extra, obj_name=pipeline_name)
            return old_init_func(self, *args, **kwargs)

        old_init_func = pipeline_cls.__init__
        pipeline_name = pipeline_cls.entities
        if isinstance(pipeline_name, list):
            assert len(pipeline_name) == 1, pipeline_name
            pipeline_name = pipeline_name[0]
        pipeline_cls.__init__ = _wrapper
        return pipeline_cls

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


def get_serving_dep_specs():
    dep_specs = []
    for item in EXTRAS["serving"].values():
        dep_specs += item
    return dep_specs


def is_paddle2onnx_plugin_available():
    return is_dep_available("paddle2onnx")


def require_paddle2onnx_plugin():
    if not is_paddle2onnx_plugin_available():
        raise RuntimeError(
            "The Paddle2ONNX plugin is not available. Please install it properly."
        )


def get_paddle2onnx_spec():
    return "paddle2onnx == 2.0.2rc3"
