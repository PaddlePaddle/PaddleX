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

from typing import Dict

from PIL import Image, ImageDraw

from ....utils.fonts import PINGFANG_FONT_FILE_PATH, create_font
from ...common.result import BaseCVResult, JsonMixin


class DocPreprocessorResult(BaseCVResult):
    """doc preprocessor result"""

    def _to_img(self) -> Dict[str, Image.Image]:
        """
        Generate an image combining the original, rotated, and unwarping images.

        Returns:
            Dict[Image.Image]: A new image combining the original, rotated, and unwarping images
        """
        image = self["input_img"][:, :, ::-1]
        rot_img = self["rot_img"][:, :, ::-1]
        angle = self["angle"]
        output_img = self["output_img"][:, :, ::-1]
        use_doc_orientation_classify = self["model_settings"][
            "use_doc_orientation_classify"
        ]
        use_doc_unwarping = self["model_settings"]["use_doc_unwarping"]
        h1, w1 = image.shape[0:2]
        h2, w2 = rot_img.shape[0:2]
        h3, w3 = output_img.shape[0:2]
        h = max(max(h1, h2), h3)
        img_show = Image.new("RGB", (w1 + w2 + w3, h + 25), (255, 255, 255))
        img_show.paste(Image.fromarray(image), (0, 0, w1, h1))
        img_show.paste(Image.fromarray(rot_img), (w1, 0, w1 + w2, h2))
        img_show.paste(Image.fromarray(output_img), (w1 + w2, 0, w1 + w2 + w3, h3))

        draw_text = ImageDraw.Draw(img_show)
        txt_list = ["Original Image", "Rotated Image", "Unwarping Image"]
        txt_list[1] = f"Rotated Image ({use_doc_orientation_classify}, {angle})"
        txt_list[2] = f"Unwarping Image ({use_doc_unwarping})"
        region_w_list = [w1, w2, w3]
        beg_w_list = [0, w1, w1 + w2]
        for tno in range(len(txt_list)):
            txt = txt_list[tno]
            font = create_font(txt, (region_w_list[tno], 20), PINGFANG_FONT_FILE_PATH)
            draw_text.text(
                [10 + beg_w_list[tno], h + 2], txt, fill=(0, 0, 0), font=font
            )
        imgs = {"preprocessed_img": img_show}
        return imgs

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            Dict[str, str]: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["model_settings"] = self["model_settings"]
        data["angle"] = self["angle"]
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> Dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            Dict[str, str]: A dictionary containing the object's data in JSON format.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["model_settings"] = self["model_settings"]
        data["angle"] = self["angle"]
        return JsonMixin._to_json(data, *args, **kwargs)
