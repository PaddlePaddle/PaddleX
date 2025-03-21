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

from .base_result import BaseResult
from .mixin import CSVMixin, ImgMixin


class BaseTSResult(BaseResult, CSVMixin, ImgMixin):
    """Base class for times series results."""

    INPUT_TS_KEY = "input_ts"

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseTSResult.

        Args:
            data (dict): The initial data.

        Raises:
            AssertionError: If the required key (`BaseTSResult.INPUT_TS_KEY`) are not found in the data.
        """
        assert (
            BaseTSResult.INPUT_TS_KEY in data
        ), f"`{BaseTSResult.INPUT_TS_KEY}` is needed, but not found in `{list(data.keys())}`!"
        data.pop("input_ts", None)

        super().__init__(data)
        CSVMixin.__init__(self, "pandas")
        ImgMixin.__init__(self, "pillow")
