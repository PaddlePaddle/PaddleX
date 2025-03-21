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

# Code copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
import importlib
import inspect
import os
import types

from . import logging
from .flags import FLAGS_json_format_model


def disable_pir_bydefault():
    # FLAGS_json_format_model not set
    if not FLAGS_json_format_model:
        os.environ["FLAGS_json_format_model"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        logging.debug("FLAGS_enable_pir_api has been set 0")


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies."""

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module = None

        super(LazyLoader, self).__init__(name)

    @property
    def loaded(self):
        return self._module is not None

    def _load(self):
        # TODO(gaotingquan): disable PIR using Flag
        if self.__name__ == "paddle":
            disable_pir_bydefault()
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self._module = module

    def __getattr__(self, item):
        logging.debug("lazy load in : %s", inspect.currentframe().f_back)
        if not self.loaded:
            # HACK: For circumventing shared library symbol conflicts when
            # importing paddlex_hpi
            if item in ("__file__",):
                raise AttributeError
            self._load()
        return getattr(self._module, item)

    def __dir__(self):
        if not self.loaded:
            self._load()
        return dir(self._module)
