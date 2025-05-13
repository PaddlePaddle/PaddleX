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

from ...utils.errors import UnsupportedAPIError
from ..base import BaseTrainer
from .model_list import MODELS


class DocVLMTrainer(BaseTrainer):
    """Document Vision Language Model Trainer"""

    entities = MODELS

    def __init__(self, config):
        # not support for now
        raise UnsupportedAPIError(
            "Document vision language models do not support train for now."
        )

    def update_config(self):
        """update training config"""

    def get_train_kwargs(self) -> dict:
        """get key-value arguments of model training function

        Returns:
            dict: the arguments of training function.
        """
        train_args = {"device": self.get_device()}
        return train_args
