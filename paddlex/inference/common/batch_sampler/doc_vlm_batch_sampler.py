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

    model_names_only_supports_batchsize_of_one = {"PP-DocBee-2B", "PP-DocBee-7B"}

    def __init__(self, model_name, batch_size: int = 1) -> None:
        """Initializes the BaseBatchSampler.

        Args:
            model_name (str): The name of the model.
            batch_size (int, optional): The size of each batch. Only support 1.
        """
        self.model_name = model_name
        if (
            self.model_name in self.model_names_only_supports_batchsize_of_one
            and batch_size != 1
        ):
            logging.warning(
                f"doc vlm batch sampler only support batch size 1 for {self.model_name}, but got {batch_size} and it will not take effect."
            )
            batch_size = 1
        super().__init__(batch_size)

    def sample(self, inputs):
        """Generate list of input file path.

        Args:
            inputs (str): file path.

        Yields:
            list: list of file path.
        """
        if isinstance(inputs, dict):
            inputs = [inputs]
        if not (isinstance(inputs, list) and all(isinstance(i, dict) for i in inputs)):
            raise TypeError(
                f"Not supported input data type! Only `Dict` or `List[Dict]` are supported, but got: {type(inputs)}."
            )

        batch = []
        for input_ in inputs:
            batch.append(input_)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch

    @BaseBatchSampler.batch_size.setter
    def batch_size(self, batch_size):
        """Sets the batch size.

        Args:
            batch_size (int): The batch size to set.

        Raises:
            Warning: If the batch size is not equal 1.
        """
        # only support batch size 1
        if (
            self.model_name in self.model_names_only_supports_batchsize_of_one
            and batch_size != 1
        ):
            logging.warning(
                f"doc vlm batch sampler only support batch size 1 for {self.model_name}, but got {batch_size} and it will not take effect."
            )
        else:
            self._batch_size = batch_size
