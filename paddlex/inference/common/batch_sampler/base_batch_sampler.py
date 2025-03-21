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

from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Tuple


class Batch:
    def __init__(self):
        self.instances = []
        self.input_paths = []

    def append(self, instance, input_path):
        self.instances.append(instance)
        self.input_paths.append(input_path)

    def reset(self):
        self.instances = []
        self.input_paths = []

    def __len__(self):
        return len(self.instances)


class BaseBatchSampler:
    """BaseBatchSampler"""

    def __init__(self, batch_size: int = 1) -> None:
        """Initializes the BaseBatchSampler.

        Args:
            batch_size (int, optional): The size of each batch. Defaults to 1.
        """
        super().__init__()
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        """Gets the batch size."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        """Sets the batch size.

        Args:
            batch_size (int): The batch size to set.

        Raises:
            AssertionError: If the batch size is not greater than 0.
        """
        assert batch_size > 0
        self._batch_size = batch_size

    def __call__(self, input: Any) -> Iterator[List[Any]]:
        """
        Sample batch data with the specified input.

        If input is None and benchmarking is enabled, it will yield batches
        of random data for the specified number of iterations.
        Otherwise, it will yield from the apply() function.

        Args:
            input (Any): The input data to sampled.

        Yields:
            Iterator[List[Any]]: An iterator yielding the batch data.
        """
        yield from self.sample(input)

    @abstractmethod
    def sample(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> Iterator[list]:
        """sample batch data"""
        raise NotImplementedError

    @abstractmethod
    def _rand_batch(self, batch_size: int) -> List[Any]:
        """rand batch data

        Args:
            batch_size (int): batch size
        """
        raise NotImplementedError
