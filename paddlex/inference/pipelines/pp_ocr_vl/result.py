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

from __future__ import annotations

from functools import partial

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT
from ...common.result import (
    BaseCVResult,
    HtmlMixin,
    JsonMixin,
    MarkdownMixin,
    XlsxMixin,
)
from ..layout_parsing.result_v2 import (
    format_centered_by_html,
    format_first_line_func,
    format_image_plain_func,
    format_image_scaled_by_html_func,
    format_text_plain_func,
    format_title_func,
    simplify_table_func,
)


class PPOCRVLBlock(object):
    """PPOCRVL Block Class"""

    def __init__(self, label, bbox, content="") -> None:
        """
        Initialize a PPOCRVLBlock object.

        Args:
            label (str): Label assigned to the block.
            bbox (list): Bounding box coordinates of the block.
            content (str, optional): Content of the block. Defaults to an empty string.
        """
        self.label = label
        self.bbox = list(map(int, bbox))
        self.content = content
        self.image = None

    def __str__(self) -> str:
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str

    def __repr__(self) -> str:
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str


def get_text_width(font, text):
    # Pillow 8.0+ 有 getlength
    if hasattr(font, "getlength"):
        return font.getlength(text)
    elif hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    else:
        return font.getsize(text)[0]


def get_text_height(font):
    # 字体高度
    if hasattr(font, "getbbox"):
        bbox = font.getbbox("A")
        return bbox[3] - bbox[1]
    else:
        return font.getsize("A")[1]


def create_image_with_text(
    image_array, result, font_path, initial_font_size=20, line_spacing=4
):
    # 将输入的 NumPy 数组转换为 PIL 图像
    image = Image.fromarray(image_array)
    image_width, image_height = image.size

    # 新图像宽度为原图两倍
    new_image_width = image_width * 2
    new_image = Image.new("RGB", (new_image_width, image_height), "white")
    new_image.paste(image, (0, 0))

    # 创建绘图对象
    draw = ImageDraw.Draw(new_image)

    # 最大文本区宽度
    max_text_width = image_width - 20  # 右侧留边

    # 自动调整字体大小以适应文本区
    font_size = initial_font_size
    while font_size > 0:
        font = ImageFont.truetype(font_path, font_size)
        # 分行：按最大宽度自动分行
        lines = []
        text = result
        while text:
            l, r = 1, len(text)
            while l <= r:
                m = (l + r) // 2
                part = text[:m]
                w = get_text_width(font, part)
                if w <= max_text_width:
                    l = m + 1
                else:
                    r = m - 1
            use_len = max(1, r)
            lines.append(text[:use_len])
            text = text[use_len:]
        # 计算总文本高度
        line_height = get_text_height(font)
        text_height = len(lines) * (line_height + line_spacing)
        if text_height <= image_height:
            break
        font_size -= 1

    # 计算文本起始坐标
    text_x = image_width + 10
    text_y = (image_height - text_height) // 2

    # 绘制文本
    for line in lines:
        draw.text((text_x, text_y), line, font=font, fill="black")
        text_y += line_height + line_spacing

    return new_image


class PPOCRVLResult(BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin):
    """Layout Parsing Result V2"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)

    def _to_img(self) -> dict[str, np.ndarray]:
        from ..layout_parsing.utils import get_show_color

        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            for key, value in self["doc_preprocessor_res"].img.items():
                res_img_dict[key] = value
        res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]

        # for layout ordering image
        image = Image.fromarray(self["doc_preprocessor_res"]["output_img"][:, :, ::-1])
        draw = ImageDraw.Draw(image, "RGBA")
        font_size = int(0.018 * int(image.width)) + 2
        font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")
        parsing_result = self["parsing_res_list"]
        for index, block in enumerate(parsing_result):
            bbox = block.bbox
            label = block.label
            fill_color = get_show_color(label, False)
            draw.rectangle(bbox, fill=fill_color)
            if index is not None:
                text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
                if int(image.width) - bbox[2] < font_size:
                    text_position = (
                        int(bbox[2] - font_size * 1.1),
                        bbox[1] - font_size // 2,
                    )
                draw.text(text_position, str(index + 1), font=font, fill="red")

        res_img_dict["layout_order_res"] = image

        for index, vl_rec in enumerate(self["vl_rec_res_list"]):
            image = vl_rec["image"]
            result = vl_rec["result"]
            image = create_image_with_text(image, result, PINGFANG_FONT.path)
            res_img_dict[f"vl_res/vl_rec_res_{index}"] = image

        return res_img_dict

    def _to_html(self) -> dict[str, str]:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        res_html_dict = {}
        if len(self["table_res_list"]) > 0:
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                table_region_id = table_res["table_region_id"]
                key = f"table_{table_region_id}"
                res_html_dict[key] = table_res.html["pred"]
        return res_html_dict

    def _to_xlsx(self) -> dict[str, str]:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            Dict[str, str]: The str type XLSX representation result.
        """
        res_xlsx_dict = {}
        if len(self["table_res_list"]) > 0:
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                table_region_id = table_res["table_region_id"]
                key = f"table_{table_region_id}"
                res_xlsx_dict[key] = table_res.xlsx["pred"]
        return res_xlsx_dict

    def _to_str(self, *args, **kwargs) -> dict[str, str]:
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
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        data["layout_det_res"] = self["layout_det_res"].str["res"]

        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> dict[str, str]:
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        parsing_res_list = self["parsing_res_list"]
        parsing_res_list = [
            {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
            }
            for parsing_res in parsing_res_list
        ]
        data["parsing_res_list"] = parsing_res_list
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        data["layout_det_res"] = self["layout_det_res"].json["res"]
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_markdown(self, pretty=True) -> dict:
        """
        Save the parsing result to a Markdown file.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.

        Returns:
            Dict
        """
        original_image_width = self["doc_preprocessor_res"]["output_img"].shape[1]

        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                )
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = format_image_plain_func

        if pretty:
            format_table_func = lambda block: "\n" + format_text_func(block).replace(
                "<table>", '<table border="1">'
            )
        else:
            format_table_func = lambda block: simplify_table_func("\n" + block.content)

        format_formula_func = lambda block: f"$${block.content}$$"
        handle_funcs_dict = {
            "paragraph_title": format_title_func,
            "abstract_title": format_title_func,
            "reference_title": format_title_func,
            "content_title": format_title_func,
            "doc_title": lambda block: f"# {block.content}".replace(
                "-\n",
                "",
            ).replace("\n", " "),
            "table_title": format_text_func,
            "figure_title": format_text_func,
            "chart_title": format_text_func,
            "vision_footnote": lambda block: block.content.replace(
                "\n\n", "\n"
            ).replace("\n", "\n\n"),
            "text": lambda block: block.content.replace("\n\n", "\n").replace(
                "\n", "\n\n"
            ),
            "abstract": partial(
                format_first_line_func,
                templates=["摘要", "abstract"],
                format_func=lambda l: f"## {l}\n",
                spliter=" ",
            ),
            "content": lambda block: block.content.replace("-\n", "  \n").replace(
                "\n", "  \n"
            ),
            "image": format_image_func,
            "chart": format_image_func,
            "formula": format_formula_func,
            "table": format_table_func,
            "reference": partial(
                format_first_line_func,
                templates=["参考文献", "references"],
                format_func=lambda l: f"## {l}",
                spliter="\n",
            ),
            "algorithm": lambda block: block.content.strip("\n"),
            "seal": format_image_func,
        }

        markdown_content = ""
        markdown_info = {}
        markdown_info["markdown_images"] = {}
        for block in self["parsing_res_list"]:
            label = block.label
            if block.image is not None:
                markdown_info["markdown_images"][block.image["path"]] = block.image[
                    "img"
                ]
            handle_func = handle_funcs_dict.get(label, None)
            if handle_func:
                markdown_content += (
                    "\n\n" + handle_func(block)
                    if markdown_content
                    else handle_func(block)
                )

        markdown_info["page_index"] = self["page_index"]
        markdown_info["input_path"] = self["input_path"]
        markdown_info["markdown_texts"] = markdown_content
        for img in self["imgs_in_doc"]:
            markdown_info["markdown_images"][img["path"]] = img["img"]

        return markdown_info
