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

from ...common.result import BaseCVResult, MarkdownMixin


class MarkdownResult(BaseCVResult, MarkdownMixin):
    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        MarkdownMixin.__init__(self)

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self.get("page_index", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            fn = f"{stem}_{page_idx}{suffix}"
        if (language := self.get("language", None)) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            fn = f"{stem}_{language}{suffix}"
        return fn

    def _to_markdown(self, pretty=True) -> dict:
        return self
