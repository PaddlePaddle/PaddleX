# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from .mixin import ImgMixin, VideoMixin


class BaseVideoResult(BaseResult, VideoMixin):
    """Base class for computer vision results."""

    INPUT_IMG_KEY = "input_img"

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseVideoResult.

        Args:
            data (dict): The initial data.

        Raises:
            AssertionError: If the required key (`BaseVideoResult.INPUT_IMG_KEY`) are not found in the data.
        """

        super().__init__(data)
        VideoMixin.__init__(self, "opencv")
