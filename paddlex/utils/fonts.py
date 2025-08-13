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

import PIL
from PIL import ImageFont

from . import logging
from .cache import CACHE_DIR
from .download import download
from .flags import LOCAL_FONT_FILE_PATH


def create_font(txt: str, sz: tuple, font_path: str) -> ImageFont:
    """
    Create a font object with specified size and path, adjusted to fit within the given image region.

    Parameters:
    txt (str): The text to be rendered with the font.
    sz (tuple): A tuple containing the height and width of an image region, used for font size.
    font_path (str): The path to the font file.

    Returns:
    ImageFont: An ImageFont object adjusted to fit within the given image region.
    """

    font_size = int(sz[1] * 0.8)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    if int(PIL.__version__.split(".")[0]) < 10:
        length = font.getsize(txt)[0]
    else:
        length = font.getlength(txt)

    if length > sz[0]:
        font_size = int(font_size * sz[0] / length)
        font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    return font


def create_font_vertical(
    txt: str, sz: tuple, font_path: str, scale=1.2
) -> ImageFont.FreeTypeFont:
    n = len(txt) if len(txt) > 0 else 1
    base_font_size = int(sz[1] / n * 0.8 * scale)
    base_font_size = max(base_font_size, 10)
    font = ImageFont.truetype(font_path, base_font_size, encoding="utf-8")

    if int(PIL.__version__.split(".")[0]) < 10:
        max_char_width = max([font.getsize(c)[0] for c in txt])
    else:
        max_char_width = max([font.getlength(c) for c in txt])

    if max_char_width > sz[0]:
        new_size = int(base_font_size * sz[0] / max_char_width)
        new_size = max(new_size, 10)
        font = ImageFont.truetype(font_path, new_size, encoding="utf-8")

    return font


class Font:
    def __init__(self, font_name=None, local_path=None):
        if local_path is None:
            if Path(str(LOCAL_FONT_FILE_PATH)).is_file():
                local_path = str(LOCAL_FONT_FILE_PATH)
        self._local_path = local_path
        if not local_path:
            assert font_name is not None
            self._font_name = font_name

    @property
    def path(self):
        # HACK: download font file when needed only
        if not self._local_path:
            self._get_offical_font()
        return self._local_path

    def _get_offical_font(self):
        """
        Download the official font file.
        """
        font_path = (Path(CACHE_DIR) / "fonts" / self._font_name).resolve().as_posix()
        if not Path(font_path).is_file():
            download(
                url=f"https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/{self._font_name}",
                save_path=font_path,
            )
        self._local_path = font_path


if Path(str(LOCAL_FONT_FILE_PATH)).is_file():
    logging.warning(
        f"Using the local font file(`{LOCAL_FONT_FILE_PATH}`) specified by `LOCAL_FONT_FILE_PATH`!"
    )

PINGFANG_FONT = Font(font_name="PingFang-SC-Regular.ttf")
SIMFANG_FONT = Font(font_name="simfang.ttf")
LATIN_FONT = Font(font_name="latin.ttf")
TH_FONT = Font(font_name="th.ttf")
EL_FONT = Font(font_name="el.ttf")
KOREAN_FONT = Font(font_name="korean.ttf")
ARABIC_FONT = Font(font_name="arabic.ttf")
CYRILLIC_FONT = Font(font_name="cyrillic.ttf")
KANNADA_FONT = Font(font_name="kannada.ttf")
TELUGU_FONT = Font(font_name="telugu.ttf")
TAMIL_FONT = Font(font_name="tamil.ttf")
DEVANAGARI_FONT = Font(font_name="devanagari.ttf")
