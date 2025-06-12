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

import copy
import re
from functools import partial
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT_FILE_PATH
from ...common.result import (
    BaseCVResult,
    HtmlMixin,
    JsonMixin,
    MarkdownMixin,
    XlsxMixin,
)
from .layout_objects import LayoutBlock
from .utils import get_seg_flag


def compile_title_pattern():
    # Precompiled regex pattern for matching numbering at the beginning of the title
    numbering_pattern = (
        r"(?:" + r"[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|" + r"[\(\（](?:[1-9][0-9]*|["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|" + r"["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+"
        r"[、\.]?|" + r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.?" + r")"
    )
    return re.compile(r"^\s*(" + numbering_pattern + r")(\s*)(.*)$")


TITLE_RE_PATTERN = compile_title_pattern()


def format_title_func(block):
    """
    Normalize chapter title.
    Add the '#' to indicate the level of the title.
    If numbering exists, ensure there's exactly one space between it and the title content.
    If numbering does not exist, return the original title unchanged.

    :param title: Original chapter title string.
    :return: Normalized chapter title string.
    """
    title = block.content
    match = TITLE_RE_PATTERN.match(title)
    if match:
        numbering = match.group(1).strip()
        title_content = match.group(3).lstrip()
        # Return numbering and title content separated by one space
        title = numbering + " " + title_content

    title = title.rstrip(".")
    level = (
        title.count(
            ".",
        )
        + 1
        if "." in title
        else 1
    )
    return f"#{'#' * level} {title}".replace("-\n", "").replace(
        "\n",
        " ",
    )


def format_centered_by_html(string):
    return (
        f'<div style="text-align: center;">{string}</div>'.replace(
            "-\n",
            "",
        ).replace("\n", " ")
        + "\n"
    )


def format_text_plain_func(block):
    return block.content


def format_image_scaled_by_html_func(block, original_image_width):
    img_tags = []
    image_path = block.image["path"]
    image_width = block.image["img"].width
    scale = int(image_width / original_image_width * 100)
    img_tags.append(
        '<img src="{}" alt="Image" width="{}%" />'.format(
            image_path.replace("-\n", "").replace("\n", " "), scale
        ),
    )
    return "\n".join(img_tags)


def format_image_plain_func(block):
    img_tags = []
    image_path = block.image["path"]
    img_tags.append("![]({})".format(image_path.replace("-\n", "").replace("\n", " ")))
    return "\n".join(img_tags)


def format_chart2table_func(block):
    lines_list = block.content.split("\n")
    column_num = len(lines_list[0].split("|"))
    lines_list.insert(1, "|".join(["---"] * column_num))
    lines_list = [f"|{line}|" for line in lines_list]
    return "\n".join(lines_list)


def simplify_table_func(table_code):
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")


def format_first_line_func(block, templates, format_func, spliter):
    lines = block.content.split(spliter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return spliter.join(lines)


class LayoutParsingResultV2(BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin):
    """Layout Parsing Result V2"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)

    def _to_img(self) -> dict[str, np.ndarray]:
        from .utils import get_show_color

        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            for key, value in self["doc_preprocessor_res"].img.items():
                res_img_dict[key] = value
        res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]

        if model_settings["use_region_detection"]:
            res_img_dict["region_det_res"] = self["region_det_res"].img["res"]

        res_img_dict["overall_ocr_res"] = self["overall_ocr_res"].img["ocr_res_img"]

        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            table_cell_img = Image.fromarray(
                copy.deepcopy(self["doc_preprocessor_res"]["output_img"][:, :, ::-1])
            )
            table_draw = ImageDraw.Draw(table_cell_img)
            rectangle_color = (255, 0, 0)
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                cell_box_list = table_res["cell_box_list"]
                for box in cell_box_list:
                    x1, y1, x2, y2 = [int(pos) for pos in box]
                    table_draw.rectangle(
                        [x1, y1, x2, y2], outline=rectangle_color, width=2
                    )
            res_img_dict["table_cell_img"] = table_cell_img

        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                seal_region_id = seal_res["seal_region_id"]
                sub_seal_res_dict = seal_res.img
                key = f"seal_res_region{seal_region_id}"
                res_img_dict[key] = sub_seal_res_dict["ocr_res_img"]

        # for layout ordering image
        image = Image.fromarray(self["doc_preprocessor_res"]["output_img"][:, :, ::-1])
        draw = ImageDraw.Draw(image, "RGBA")
        font_size = int(0.018 * int(image.width)) + 2
        font = ImageFont.truetype(PINGFANG_FONT_FILE_PATH, font_size, encoding="utf-8")
        parsing_result: List[LayoutBlock] = self["parsing_res_list"]
        for block in parsing_result:
            bbox = block.bbox
            index = block.order_index
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
                draw.text(text_position, str(index), font=font, fill="red")

        res_img_dict["layout_order_res"] = image

        return res_img_dict

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
        data["overall_ocr_res"] = self["overall_ocr_res"].str["res"]
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            data["table_res_list"] = []
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                data["table_res_list"].append(table_res.str["res"])
        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            data["seal_res_list"] = []
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                data["seal_res_list"].append(seal_res.str["res"])
        if (
            model_settings["use_formula_recognition"]
            and len(self["formula_res_list"]) > 0
        ):
            data["formula_res_list"] = []
            for sno in range(len(self["formula_res_list"])):
                formula_res = self["formula_res_list"][sno]
                data["formula_res_list"].append(formula_res.str["res"])

        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> dict[str, str]:
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
        data["overall_ocr_res"] = self["overall_ocr_res"].json["res"]
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            data["table_res_list"] = []
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                data["table_res_list"].append(table_res.json["res"])
        if model_settings["use_seal_recognition"] and len(self["seal_res_list"]) > 0:
            data["seal_res_list"] = []
            for sno in range(len(self["seal_res_list"])):
                seal_res = self["seal_res_list"][sno]
                data["seal_res_list"].append(seal_res.json["res"])
        if (
            model_settings["use_formula_recognition"]
            and len(self["formula_res_list"]) > 0
        ):
            data["formula_res_list"] = []
            for sno in range(len(self["formula_res_list"])):
                formula_res = self["formula_res_list"][sno]
                data["formula_res_list"].append(formula_res.json["res"])
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_html(self) -> dict[str, str]:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        model_settings = self["model_settings"]
        res_html_dict = {}
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
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
        model_settings = self["model_settings"]
        res_xlsx_dict = {}
        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            for sno in range(len(self["table_res_list"])):
                table_res = self["table_res_list"][sno]
                table_region_id = table_res["table_region_id"]
                key = f"table_{table_region_id}"
                res_xlsx_dict[key] = table_res.xlsx["pred"]
        return res_xlsx_dict

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

        if self["model_settings"].get("use_chart_recognition", False):
            format_chart_func = format_chart2table_func
        else:
            format_chart_func = format_image_func

        if self["model_settings"].get("use_seal_recognition", False):
            format_seal_func = lambda block: "\n".join(
                [format_image_func(block), format_text_func(block)]
            )
        else:
            format_seal_func = format_image_func

        if self["model_settings"].get("use_table_recognition", False):
            if pretty:
                format_table_func = lambda block: "\n" + format_text_func(
                    block
                ).replace("<table>", '<table border="1">')
            else:
                format_table_func = lambda block: simplify_table_func(
                    "\n" + block.content
                )
        else:
            format_table_func = format_image_func

        if self["model_settings"].get("use_formula_recognition", False):
            format_formula_func = lambda block: f"$${block.content}$$"
        else:
            format_formula_func = format_image_func

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
            "chart": format_chart_func,
            "formula": format_formula_func,
            "table": format_table_func,
            "reference": partial(
                format_first_line_func,
                templates=["参考文献", "references"],
                format_func=lambda l: f"## {l}",
                spliter="\n",
            ),
            "algorithm": lambda block: block.content.strip("\n"),
            "seal": format_seal_func,
        }

        markdown_content = ""
        last_label = None
        seg_start_flag = None
        seg_end_flag = None
        prev_block = None
        page_first_element_seg_start_flag = None
        page_last_element_seg_end_flag = None
        markdown_info = {}
        markdown_info["markdown_images"] = {}
        for block in self["parsing_res_list"]:
            seg_start_flag, seg_end_flag = get_seg_flag(block, prev_block)

            label = block.label
            if block.image is not None:
                markdown_info["markdown_images"][block.image["path"]] = block.image[
                    "img"
                ]
            page_first_element_seg_start_flag = (
                seg_start_flag
                if (page_first_element_seg_start_flag is None)
                else page_first_element_seg_start_flag
            )

            handle_func = handle_funcs_dict.get(label, None)
            if handle_func:
                prev_block = block
                if label == last_label == "text" and seg_start_flag == False:
                    markdown_content += handle_func(block)
                else:
                    markdown_content += (
                        "\n\n" + handle_func(block)
                        if markdown_content
                        else handle_func(block)
                    )
                last_label = label
        page_last_element_seg_end_flag = seg_end_flag

        markdown_info["markdown_texts"] = markdown_content
        markdown_info["page_continuation_flags"] = (
            page_first_element_seg_start_flag,
            page_last_element_seg_end_flag,
        )
        for img in self["imgs_in_doc"]:
            markdown_info["markdown_images"][img["path"]] = img["img"]

        return markdown_info
