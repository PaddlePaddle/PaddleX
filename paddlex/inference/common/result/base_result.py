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

import inspect
import random
import time
from pathlib import Path

import numpy as np

from ....utils import logging
from .mixin import JsonMixin, StrMixin


class BaseResult(dict, JsonMixin, StrMixin):
    """Base class for result objects that can save themselves.

    This class inherits from dict and provides properties and methods for handling result.
    """

    def __init__(self, data: dict) -> None:
        """Initializes the BaseResult with the given data.

        Args:
            data (dict): The initial data.
        """
        super().__init__(data)
        self._save_funcs = []
        StrMixin.__init__(self)
        JsonMixin.__init__(self)
        np.set_printoptions(threshold=1, edgeitems=1)
        self._rand_fn = None

    def save_all(self, save_path: str) -> None:
        """Calls all registered save methods with the given save path.

        Args:
            save_path (str): The path to save the result to.
        """
        for func in self._save_funcs:
            signature = inspect.signature(func)
            if "save_path" in signature.parameters:
                func(save_path=save_path)
            else:
                func()

    def _get_input_fn(self):
        if self.get("input_path", None) is None:
            if self._rand_fn:
                return self._rand_fn

            timestamp = int(time.time())
            random_number = random.randint(1000, 9999)
            fp = f"{timestamp}_{random_number}"
            logging.warning(
                f"There is not input file name as reference for name of saved result file. So the saved result file would be named with timestamp and random number: `{fp}`."
            )
            self._rand_fn = Path(fp).name
            return self._rand_fn
        fp = self["input_path"]
        return Path(fp).name
