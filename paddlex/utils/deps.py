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

_SUPPORTED_GENAI_ENGINE_BACKENDS = ["fastdeploy-server", "vllm-server", "sglang-server"]


class DependencyError(Exception):
    pass


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


def _get_base_dep_specs(required_only=False):
    dep_specs = defaultdict(list)
    for dep_spec in importlib.metadata.requires("paddlex"):
        extra_name, dep_spec = _get_extra_name_and_remove_extra_marker(dep_spec)
        if (required_only and extra_name is None) or (
            not required_only and (extra_name is None or extra_name == "base")
        ):
            dep_spec = dep_spec.rstrip()
            req = Requirement(dep_spec)
            dep_specs[req.name].append(dep_spec)
    return dep_specs


BASE_DEP_SPECS = _get_base_dep_specs()
REQUIRED_DEP_SPECS = _get_base_dep_specs(required_only=True)


def get_dep_version(dep):
    try:
        return importlib.metadata.version(dep)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache()
def is_dep_available(dep, /, check_version=False):
    if (
        dep in ("paddlepaddle", "paddle-custom-device", "ultra-infer", "fastdeploy")
        and check_version
    ):
        raise ValueError(
            "`check_version` is not allowed to be `True` for `paddlepaddle`, `paddle-custom-device`, `ultra-infer`, and `fastdeploy`."
        )
    # Currently for several special deps we check if the import packages exist.
    if dep == "paddlepaddle":
        return importlib.util.find_spec("paddle") is not None
    elif dep == "paddle-custom-device":
        return importlib.util.find_spec("paddle_custom_device") is not None
    elif dep == "ultra-infer":
        return importlib.util.find_spec("ultra_infer") is not None
    elif dep == "fastdeploy":
        return importlib.util.find_spec("fastdeploy") is not None
    version = get_dep_version(dep)
    if version is None:
        return False
    if check_version:
        if dep not in BASE_DEP_SPECS:
            raise ValueError(
                f"Currently, `check_version=True` is supported only for base dependencies."
            )
        for dep_spec in BASE_DEP_SPECS[dep]:
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
        raise DependencyError(msg)


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


def require_extra(extra, *, obj_name=None, alt=None):
    if is_extra_available(extra) or (alt is not None and is_extra_available(alt)):
        return
    if obj_name is not None:
        msg = f"`{obj_name}` requires additional dependencies."
    else:
        msg = "Additional dependencies are required."
    msg += f' To install them, run `pip install "paddlex[{extra}]==<PADDLEX_VERSION>"` if you’re installing `paddlex` from an index, or `pip install -e "/path/to/PaddleX[{extra}]"` if you’re installing `paddlex` locally.'
    if alt is not None:
        msg += f" Alternatively, you can install `paddlex[{alt}]` instead."
    raise DependencyError(msg)


def pipeline_requires_extra(extra, *, alt=None):
    def _deco(pipeline_cls):
        @wraps(pipeline_cls.__init__)
        def _wrapper(self, *args, **kwargs):
            require_extra(extra, obj_name=pipeline_name, alt=alt)
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
        raise DependencyError(
            "The high-performance inference plugin is not available. Please install it properly."
        )


def is_serving_plugin_available():
    return is_extra_available("serving")


def require_serving_plugin():
    if not is_serving_plugin_available():
        raise DependencyError(
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
        raise DependencyError(
            "The Paddle2ONNX plugin is not available. Please install it properly."
        )


def get_paddle2onnx_dep_specs():
    dep_specs = []
    for item in EXTRAS["paddle2onnx"].values():
        dep_specs += item
    return dep_specs


def is_genai_engine_plugin_available(backend="any"):
    if backend != "any" and backend not in _SUPPORTED_GENAI_ENGINE_BACKENDS:
        raise ValueError(f"Unknown backend type: {backend}")
    if backend == "any":
        for be in _SUPPORTED_GENAI_ENGINE_BACKENDS:
            if is_genai_engine_plugin_available(be):
                return True
        return False
    else:
        if "fastdeploy" in backend:
            return is_dep_available("fastdeploy")
        elif is_extra_available(f"genai-{backend}"):
            from .env import is_cuda_available

            if is_cuda_available():
                return is_dep_available("xformers") and is_dep_available("flash-attn")
            return True
        return False


def require_genai_engine_plugin(backend="any"):
    if not is_genai_engine_plugin_available(backend):
        if backend == "any":
            prefix = "The generative AI engine plugins are"
        else:
            prefix = f"The generative AI {repr(backend)} engine plugin is"
        raise RuntimeError(f"{prefix} not available. Please install it properly.")


def is_genai_client_plugin_available():
    return is_extra_available("genai-client")


def require_genai_client_plugin():
    if not is_genai_client_plugin_available():
        raise RuntimeError(
            "The generative AI client plugin is not available. Please install it properly."
        )


def get_genai_fastdeploy_spec(device_type):
    SUPPORTED_DEVICE_TYPES = ("gpu",)
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(f"Unsupported device type: {device_type}")
    if device_type == "gpu":
        return "fastdeploy-gpu == 2.3.0"
    else:
        raise AssertionError


def get_genai_dep_specs(type):
    if type != "client" and type not in _SUPPORTED_GENAI_ENGINE_BACKENDS:
        raise ValueError(f"Invalid type: {type}")
    if "fastdeploy" in type:
        raise ValueError(f"{repr(type)} is not supported")

    dep_specs = []
    for item in EXTRAS[f"genai-{type}"].values():
        dep_specs += item
    return dep_specs
