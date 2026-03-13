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

"""LatexConverter — placeholder for future LaTeX conversion implementation.

The current LaTeX conversion logic lives in ``mixin.py`` (``LatexMixin``) and
each Result class's ``_to_latex()`` method.  This module will be filled in
when the Word/LaTeX conversion path is rewritten.
"""

from __future__ import annotations


class LatexConverter:
    """Convert document blocks to a LaTeX document.

    TODO: Implement when the Word/LaTeX conversion rewrite is done.
    """

    @staticmethod
    def convert(blocks, *, original_image_width=None, **kwargs):
        """Convert *blocks* to LaTeX-compatible structures.

        Raises:
            NotImplementedError: Always — use ``_to_latex()`` on Result classes.
        """
        raise NotImplementedError(
            "LatexConverter.convert() is not yet implemented. "
            "Use the existing _to_latex() method on Result classes."
        )
