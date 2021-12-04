# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import importlib

from . import topk

from .topk import Topk, MultiLabelTopk


def build_postprocess(config):
    config = copy.deepcopy(config)
    model_name = config.pop("name")
    mod = importlib.import_module(__name__)
    postprocess_func = getattr(mod, model_name)(**config)
    return postprocess_func


class DistillationPostProcess(object):
    def __init__(self, model_name="Student", key=None, func="Topk", **kargs):
        super().__init__()
        self.func = eval(func)(**kargs)
        self.model_name = model_name
        self.key = key

    def __call__(self, x, file_names=None):
        x = x[self.model_name]
        if self.key is not None:
            x = x[self.key]
        return self.func(x, file_names=file_names)
