# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from pathlib import Path
from PIL import Image, ImageDraw

import re
import numpy as np
from PIL import Image
from PIL import ImageDraw

from ...common.result import (
    BaseCVResult,
    HtmlMixin,
    JsonMixin,
    MarkdownMixin,
    XlsxMixin,
)
from .utils import get_layout_ordering
from .utils import recursive_img_array2path
from .utils import get_show_color


class LayoutParsingResultV2(BaseCVResult, HtmlMixin, XlsxMixin, MarkdownMixin):
    """Layout Parsing Result V2"""

    def __init__(self, data) -> None:
        """Initializes a new instance of the class with the specified data."""
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)
        MarkdownMixin.__init__(self)
        JsonMixin.__init__(self)

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self["page_index"]) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            return f"{stem}_{page_idx}{suffix}"
        else:
            return fn

    def _to_img(self) -> dict[str, np.ndarray]:
        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            for key, value in self["doc_preprocessor_res"].img.items():
                res_img_dict[key] = value
        res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]

        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
            res_img_dict["overall_ocr_res"] = self["overall_ocr_res"].img["ocr_res_img"]

        if model_settings["use_table_recognition"] and len(self["table_res_list"]) > 0:
            table_cell_img = Image.fromarray(
                copy.deepcopy(self["doc_preprocessor_res"]["output_img"])
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
        image = Image.fromarray(self["doc_preprocessor_res"]["output_img"])
        draw = ImageDraw.Draw(image, "RGBA")
        parsing_result = self["parsing_res_list"]
        for block in parsing_result:
            bbox = block["block_bbox"]
            index = block.get("index", None)
            label = block["sub_label"]
            fill_color = get_show_color(label)
            draw.rectangle(bbox, fill=fill_color)
            if index is not None:
                text_position = (bbox[2] + 2, bbox[1] - 10)
                draw.text(text_position, str(index), fill="red")

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
        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
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
                "block_label": parsing_res["block_label"],
                "block_content": parsing_res["block_content"],
                "block_bbox": parsing_res["block_bbox"],
            }
            for parsing_res in parsing_res_list
        ]
        data["parsing_res_list"] = parsing_res_list
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        data["layout_det_res"] = self["layout_det_res"].json["res"]
        if model_settings["use_general_ocr"] or model_settings["use_table_recognition"]:
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

    def _to_markdown(self) -> dict:
        """
        Save the parsing result to a Markdown file.

        Returns:
            Dict
        """
        recursive_img_array2path(self["parsing_res_list"], labels=["block_image"])

        def _format_data(obj):

            def format_title(content_value):
                content_value = content_value.rstrip(".")
                level = (
                    content_value.count(
                        ".",
                    )
                    + 1
                    if "." in content_value
                    else 1
                )
                return f"{'#' * level} {content_value}".replace("-\n", "").replace(
                    "\n",
                    " ",
                )

            def format_centered_text(key):
                return (
                    f'<div style="text-align: center;">{block[key]}</div>'.replace(
                        "-\n",
                        "",
                    ).replace("\n", " ")
                    + "\n"
                )

            def format_image(label):
                img_tags = []
                image_path = "".join(block[label].keys())
                img_tags.append(
                    '<div style="text-align: center;"><img src="{}" alt="Image" /></div>'.format(
                        image_path.replace("-\n", "").replace("\n", " "),
                    ),
                )
                return "\n".join(img_tags)

            def format_reference():
                pattern = r"\s*\[\s*\d+\s*\]\s*"
                res = re.sub(
                    pattern,
                    lambda match: "\n" + match.group(),
                    block["reference"].replace("\n", ""),
                )
                return "\n" + res

            def format_table():
                return "\n" + block["block_content"]

            handlers = {
                "paragraph_title": lambda: format_title(block["block_content"]),
                "doc_title": lambda: f"# {block['block_content']}".replace(
                    "-\n",
                    "",
                ).replace("\n", " "),
                "table_title": lambda: format_centered_text("block_content"),
                "figure_title": lambda: format_centered_text("block_content"),
                "chart_title": lambda: format_centered_text("block_content"),
                "text": lambda: block["block_content"]
                .replace("-\n", " ")
                .replace("\n", " "),
                "abstract": lambda: block["block_content"]
                .replace("-\n", " ")
                .replace("\n", " "),
                "content": lambda: block["block_content"]
                .replace("-\n", " ")
                .replace("\n", " "),
                "image": lambda: format_image("block_image"),
                "chart": lambda: format_image("block_image"),
                "formula": lambda: f"$${block['block_content']}$$",
                "table": format_table,
                "reference": lambda: block["block_content"],
                "algorithm": lambda: block["block_content"].strip("\n"),
                "seal": lambda: format_image("block_content"),
            }
            parsing_res_list = obj["parsing_res_list"]
            markdown_content = ""
            last_label = None
            seg_start_flag = None
            seg_end_flag = None
            for block in sorted(
                parsing_res_list,
                key=lambda x: x.get("sub_index", 999),
            ):
                label = block.get("block_label")
                seg_start_flag = block.get("seg_start_flag")
                handler = handlers.get(label)
                if handler:
                    if (
                        label == last_label == "text"
                        and seg_start_flag == seg_end_flag == False
                    ):
                        markdown_content += " " + handler()
                    else:
                        markdown_content += "\n\n" + handler()
                    last_label = label
                    seg_end_flag = block.get("seg_end_flag")

            return markdown_content

        markdown_info = dict()
        markdown_info["markdown_texts"] = _format_data(self)
        markdown_info["markdown_images"] = dict()
        for block in self["parsing_res_list"]:
            if block["block_label"] in ["image", "chart"]:
                image_path, image_value = next(iter(block["block_image"].items()))
                markdown_info["markdown_images"][image_path] = image_value

        return markdown_info
