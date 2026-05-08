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
from copy import deepcopy
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ....utils.fonts import PINGFANG_FONT
from ...common.result import (
    BaseCVResult,
    HtmlMixin,
    JsonMixin,
    LatexMixin,
    MarkdownMixin,
    WordMixin,
    XlsxMixin,
)
from ...common.result.converter import MarkdownConverter
from ...common.result.converter.markdown_format_funcs import (
    build_handle_funcs_dict,
    format_centered_by_html,
    format_chart2markdown_table,
    format_image_plain,
    format_image_scaled_by_html,
    format_text_plain,
    simplify_table,
)
from .layout_objects import LayoutBlock
from .utils import get_seg_flag


class LayoutParsingResultV2(
    BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin, WordMixin, LatexMixin
):
    """Layout Parsing Result V2"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)
        WordMixin.__init__(self)
        LatexMixin.__init__(self)

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
        font = ImageFont.truetype(PINGFANG_FONT.path, font_size, encoding="utf-8")
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
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        parsing_res_list: List[LayoutBlock] = self["parsing_res_list"]
        parsing_res_list = [
            {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
                "block_id": parsing_res.index,
                "block_order": parsing_res.order_index,
            }
            for parsing_res in parsing_res_list
        ]
        data["parsing_res_list"] = parsing_res_list
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

    def _build_handle_funcs_dict(self, pretty=True):
        """Build label-to-handler mapping for content formatting."""
        original_image_width = self["doc_preprocessor_res"]["output_img"].shape[1]
        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html(
                    block,
                    original_image_width=original_image_width,
                )
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = format_image_plain

        if self["model_settings"].get("use_chart_recognition", False):
            format_chart_func = format_chart2markdown_table
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
                format_table_func = lambda block: simplify_table("\n" + block.content)
        else:
            format_table_func = format_image_func

        if self["model_settings"].get("use_formula_recognition", False):
            format_formula_func = lambda block: f"$${block.content}$$"
        else:
            format_formula_func = format_image_func

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
            use_plain_header_footer_image=True,
        )
        for label in self["model_settings"].get("markdown_ignore_labels", []):
            handle_funcs_dict.pop(label, None)
        return handle_funcs_dict

    def _to_json(self, *args, **kwargs) -> dict[str, str]:
        """
        Converts the object's data to a JSON dictionary.

        Args:
            *args: Positional arguments passed to the JsonMixin._to_json method.
            **kwargs: Keyword arguments passed to the JsonMixin._to_json method.

        Returns:
            Dict[str, str]: A dictionary containing the object's data in JSON format.
        """
        if self["model_settings"].get("format_block_content", False):
            handle_funcs_dict = self._build_handle_funcs_dict(pretty=True)

        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        parsing_res_list: List[LayoutBlock] = self["parsing_res_list"]
        parsing_res_list_json = []
        for parsing_res in parsing_res_list:
            res_dict = {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
                "block_id": parsing_res.index,
                "block_order": parsing_res.order_index,
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

    def _to_markdown(self, pretty=True, show_formula_number=False) -> dict:
        """
        Save the parsing result to a Markdown file.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.

        Returns:
            Dict
        """
        handle_funcs_dict = self._build_handle_funcs_dict(pretty=pretty)

        result = MarkdownConverter.convert(
            self["parsing_res_list"],
            handle_funcs_dict=handle_funcs_dict,
            show_formula_number=show_formula_number,
            use_seg_flag=True,
            get_seg_flag_func=get_seg_flag,
            imgs_in_doc=self["imgs_in_doc"],
        )
        result["page_index"] = self["page_index"]
        result["input_path"] = self["input_path"]
        return result

    def _to_word(self) -> dict:
        """Convert the object's parsing result into a Word-compatible dict.

        Returns:
            dict: {
                "word_blocks": List[Dict],       # Simplified list of content blocks
                "original_image_width": int,   # Pixel width of the source page
                "input_path": str,             # Original input file path
                "images": List[Dict]           # List of {"path": str, "img": PIL.Image}
            }
        """
        from ...common.result.converter import build_word_blocks

        word_blocks, images = build_word_blocks(
            self["parsing_res_list"],
            imgs_in_doc=self.get("imgs_in_doc", []),
        )

        return {
            "word_blocks": word_blocks,
            "original_image_width": self["doc_preprocessor_res"]["output_img"].shape[1],
            "input_path": self["input_path"],
            "images": images,
        }

    def _to_latex(self) -> dict:
        """
        Convert the object's parsing result into a latex-compatible dict.

        Returns:
            dict: {
                "latex_blocks": List[Dict],       # Simplified list of content blocks
                "input_path": str,             # Original input file path
                "images": List[Dict]           # List of {"path": str, "img": PIL.Image}
            }
        """
        latex_blocks = []
        image = []

        for block in self["parsing_res_list"]:

            label = block.label
            content = getattr(block, "content", "")
            if label in ["image", "seal"]:
                if block.image is None:
                    continue
                content = block.image["path"]
            elif label == "chart":
                if block.image is not None:
                    content = block.image["path"]
                elif content:
                    # VLM chart recognition: pipe-delimited table text → reuse table rendering
                    content = content.replace("|", "\t")
                    label = "table"
                else:
                    continue
            block_dict = {
                "type": label,
                "content": deepcopy(content),
            }
            latex_blocks.append(block_dict)
            if block.image is not None:
                image.append({"path": block.image["path"], "img": block.image["img"]})

        return {
            "latex_blocks": latex_blocks,
            "images": image,
            "input_path": self["input_path"],
        }
