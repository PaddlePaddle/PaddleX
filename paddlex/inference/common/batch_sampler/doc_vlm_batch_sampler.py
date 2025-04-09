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


from ....utils import logging
from .base_batch_sampler import BaseBatchSampler


class DocVLMBatchSampler(BaseBatchSampler):
    def __init__(self):
        """Initializes the BaseBatchSampler.

        Args:
            batch_size (int, optional): The size of each batch. Only support 1.
        """
        super().__init__()
        self.batch_size = 1

    def sample(self, inputs):
        """Generate list of input file path.

        Args:
            inputs (str): file path.

        Yields:
            list: list of file path.
        """
        if isinstance(inputs, dict):
            yield [inputs]
        else:
            logging.warning(
                f"Not supported input data type! Only `dict` are supported, but got: {input}."
            )

    @BaseBatchSampler.batch_size.setter
    def batch_size(self, batch_size):
        """Sets the batch size.

        Args:
            batch_size (int): The batch size to set.

        Raises:
            Warning: If the batch size is not equal 1.
        """
        # only support batch size 1
        if batch_size != 1:
            logging.warning(
                f"doc vlm batch sampler only support batch size 1, but got {batch_size}."
            )
        else:
            self._batch_size = batch_size
