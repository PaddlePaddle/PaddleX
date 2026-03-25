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
from typing import Any, Dict, List

from .latex_converter import LatexConverter
from .markdown_converter import MarkdownConverter
from .word_converter import WordConverter, build_word_blocks


def save_images(image_list: List[Dict[str, Any]], base_save_path) -> Dict[str, str]:
    """Save images to disk, return {original_path: absolute_saved_path}.

    Args:
        image_list: List of dicts with "path" and "img" keys.
        base_save_path: Base directory; images are saved under imgs/ sub-dir.

    Returns:
        Dict mapping original image path to the absolute path where it was saved.
    """
    abs_image_paths: Dict[str, str] = {}
    base_save_path = Path(base_save_path)
    image_dir = base_save_path / "imgs"
    image_dir.mkdir(parents=True, exist_ok=True)

    for item in image_list:
        img_path = item.get("path")
        img_obj = item.get("img")
        if not img_path or not img_obj:
            continue

        img_name = Path(img_path).name
        save_path = image_dir / img_name
        img_obj.save(save_path)
        abs_image_paths[img_path] = str(save_path.resolve())
    return abs_image_paths
