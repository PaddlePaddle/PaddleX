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

from pathlib import Path

from .base_result import BaseResult
from .mixin import ImgMixin


class BaseCVResult(BaseResult, ImgMixin):
    """Base class for computer vision results."""

    def __init__(self, data: dict) -> None:
        """
        Initialize the BaseCVResult.

        Args:
            data (dict): The initial data.
        """
        super().__init__(data)
        ImgMixin.__init__(self, "pillow")

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self.get("page_index", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            return f"{stem}_{page_idx}{suffix}"
        else:
            return fn
