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

from ..base import BaseExportor
from .model_list import MODELS


class SegExportor(BaseExportor):
    """Semantic Segmentation Model Exportor"""

    entities = MODELS

    def get_export_kwargs(self):
        """get key-value arguments of model export function"""
        kwargs = super().get_export_kwargs()

        input_shape = self.export_config.get("input_shape")
        if input_shape is not None:
            kwargs["input_shape"] = input_shape
        return kwargs
