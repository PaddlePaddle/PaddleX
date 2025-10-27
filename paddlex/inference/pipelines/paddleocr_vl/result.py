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

VISUALIZE_INDEX_LABELS = [
    "text",
    "formula",
    "inline_formula",
    "display_formula",
    "algorithm",
    "reference",
    "reference_content",
    "content",
    "abstract",
    "paragraph_title",
    "doc_title",
    "vertical_text",
    "ocr",
]


class PaddleOCRVLBlock(object):
    """PaddleOCRVL Block Class"""

    def __init__(self, label, bbox, content="") -> None:
        """
        Initialize a PaddleOCRVLBlock object.

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
        """
        Return a string representation of the block.
        """
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str

    def __repr__(self) -> str:
        """
        Return a string representation of the block.
        """
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str


def merge_formula_and_number(formula, formula_number):
    """
    Merge a formula and its formula number for display.

    Args:
        formula (str): The formula string.
        formula_number (str): The formula number string.

    Returns:
        str: The merged formula with tag.
    """
    formula = formula.replace("$$", "")
    merge_formula = r"{} \tag*{{{}}}".format(formula, formula_number)
    return f"$${merge_formula}$$"


def format_chart2table_func(block):
    lines_list = block.content.split("\n")
    # 提取表头和内容
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
    # 构造HTML表格
    html = "<table border=1 style='margin: auto; width: max-content;'>\n"
    html += (
        "  <thead><tr>"
        + "".join(
            f"<th style='text-align: center;'>{cell.strip()}</th>" for cell in header
        )
        + "</tr></thead>\n"
    )
    html += "  <tbody>\n"
    for row in rows:
        html += (
            "    <tr>"
            + "".join(
                f"<td style='text-align: center;'>{cell.strip()}</td>" for cell in row
            )
            + "</tr>\n"
        )
    html += "  </tbody>\n"
    html += "</table>"
    return html


def format_table_center_func(block):
    tabel_content = block.content
    tabel_content = tabel_content.replace(
        "<table>", "<table border=1 style='margin: auto; width: max-content;'>"
    )
    tabel_content = tabel_content.replace("<th>", "<th style='text-align: center;'>")
    tabel_content = tabel_content.replace("<td>", "<td style='text-align: center;'>")
    return tabel_content


def build_handle_funcs_dict(
    *,
    text_func,
    image_func,
    chart_func,
    table_func,
    formula_func,
    seal_func,
):
    """
    Build a dictionary mapping block labels to their formatting functions.

    Args:
        text_func: Function to format text blocks.
        image_func: Function to format image blocks.
        chart_func: Function to format chart blocks.
        table_func: Function to format table blocks.
        formula_func: Function to format formula blocks.
        seal_func: Function to format seal blocks.

    Returns:
        dict: A mapping from block label to handler function.
    """
    return {
        "paragraph_title": format_title_func,
        "abstract_title": format_title_func,
        "reference_title": format_title_func,
        "content_title": format_title_func,
        "doc_title": lambda block: f"# {block.content}".replace("-\n", "").replace(
            "\n", " "
        ),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "text": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "ocr": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "vertical_text": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "reference_content": lambda block: block.content.replace("\n\n", "\n").replace(
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
        "image": image_func,
        "chart": chart_func,
        "formula": formula_func,
        "display_formula": formula_func,
        "inline_formula": formula_func,
        "table": table_func,
        "reference": partial(
            format_first_line_func,
            templates=["参考文献", "references"],
            format_func=lambda l: f"## {l}",
            spliter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": seal_func,
    }


class PaddleOCRVLResult(BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin):
    """
    PaddleOCRVLResult class for holding and formatting OCR/VL parsing results.
    """

    def __init__(self, data) -> None:
        """
        Initializes a new instance of the class with the specified data.

        Args:
            data: The input data for the parsing result.
        """
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)

    def _to_img(self) -> dict[str, np.ndarray]:
        """
        Convert the parsing result to a dictionary of images.

        Returns:
            dict: Keys are names, values are numpy arrays (images).
        """
        from ..layout_parsing.utils import get_show_color

        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            for key, value in self["doc_preprocessor_res"].img.items():
                res_img_dict[key] = value
        if self["model_settings"]["use_layout_detection"]:
            res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]

        # for layout ordering image
        image = Image.fromarray(self["doc_preprocessor_res"]["output_img"][:, :, ::-1])
        draw = ImageDraw.Draw(image, "RGBA")
        font_size = int(0.018 * int(image.width)) + 2
        font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")
        parsing_result = self["parsing_res_list"]
        order_index = 0
        for block in parsing_result:
            bbox = block.bbox
            label = block.label
            fill_color = get_show_color(label, False)
            draw.rectangle(bbox, fill=fill_color)
            if label in VISUALIZE_INDEX_LABELS:
                text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
                if int(image.width) - bbox[2] < font_size:
                    text_position = (
                        int(bbox[2] - font_size * 1.1),
                        bbox[1] - font_size // 2,
                    )
                draw.text(text_position, str(order_index + 1), font=font, fill="red")
                order_index += 1

        res_img_dict["layout_order_res"] = image

        return res_img_dict

    def _to_html(self) -> dict[str, str]:
        """
        Converts the prediction to its corresponding HTML representation.

        Returns:
            dict: The str type HTML representation result.
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
        """
        Converts the prediction HTML to an XLSX file path.

        Returns:
            dict: The str type XLSX representation result.
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
        """
        Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            dict: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        if self["model_settings"]["use_layout_detection"]:
            data["layout_det_res"] = self["layout_det_res"].str["res"]
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
        return JsonMixin._to_str(data, *args, **kwargs)

    def _to_json(self, *args, **kwargs) -> dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            dict: A dictionary containing the object's data in JSON format.
        """
        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"].get("format_block_content", False):
            original_image_width = self["doc_preprocessor_res"]["output_img"].shape[1]
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                )
            )

            if self["model_settings"].get("use_chart_recognition", False):
                format_chart_func = format_chart2table_func
            else:
                format_chart_func = format_image_func

            format_seal_func = format_image_func

            format_table_func = lambda block: "\n" + format_table_center_func(block)
            format_formula_func = lambda block: block.content

            handle_funcs_dict = build_handle_funcs_dict(
                text_func=format_text_func,
                image_func=format_image_func,
                chart_func=format_chart_func,
                table_func=format_table_func,
                formula_func=format_formula_func,
                seal_func=format_seal_func,
            )

        parsing_res_list = self["parsing_res_list"]
        parsing_res_list_json = []
        order_index = 1
        for idx, parsing_res in enumerate(parsing_res_list):
            label = parsing_res.label
            if label in VISUALIZE_INDEX_LABELS:
                order = order_index
                order_index += 1
            else:
                order = None
            res_dict = {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
                "block_id": idx,
                "block_order": order,
            }
            if self["model_settings"].get("format_block_content", False):
                if handle_funcs_dict.get(parsing_res.label):
                    res_dict["block_content"] = handle_funcs_dict[parsing_res.label](
                        parsing_res
                    )
                else:
                    res_dict["block_content"] = parsing_res.content

            parsing_res_list_json.append(res_dict)
        data["parsing_res_list"] = parsing_res_list_json
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        if self["model_settings"]["use_layout_detection"]:
            data["layout_det_res"] = self["layout_det_res"].json["res"]
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_markdown(self, pretty=True, show_formula_number=False) -> dict:
        """
        Save the parsing result to a Markdown file.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.
            show_formula_number (bool): whether to show formula numbers.

        Returns:
            dict: Markdown information with text and images.
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

        format_chart_func = (
            format_chart2table_func
            if self["model_settings"]["use_chart_recognition"]
            else format_image_func
        )

        if pretty:
            format_table_func = lambda block: "\n" + format_table_center_func(block)
        else:
            format_table_func = lambda block: simplify_table_func("\n" + block.content)

        format_formula_func = lambda block: block.content
        format_seal_func = format_image_func

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )

        markdown_content = ""
        markdown_info = {}
        markdown_info["markdown_images"] = {}
        for idx, block in enumerate(self["parsing_res_list"]):
            label = block.label
            if block.image is not None:
                markdown_info["markdown_images"][block.image["path"]] = block.image[
                    "img"
                ]
            handle_func = handle_funcs_dict.get(label, None)
            if (
                show_formula_number
                and (label == "display_formula" or label == "formula")
                and idx != len(self["parsing_res_list"]) - 1
            ):
                next_block = self["parsing_res_list"][idx + 1]
                next_block_label = next_block.label
                if next_block_label == "formula_number":
                    block.content = merge_formula_and_number(
                        block.content, next_block.content
                    )
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
