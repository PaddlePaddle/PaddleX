# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import importlib


class _LazyModule(object):
    def __init__(self, mod_name):
        super().__init__()
        self.mod_name = mod_name
        self._mod = None

    def __getattr__(self, name):
        if not self._mod:
            self._mod = importlib.import_module(self.mod_name)
        return getattr(self._mod, name)


pb_utils = _LazyModule("triton_python_backend_utils")
