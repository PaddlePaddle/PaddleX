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

import random
from functools import partial

import numpy as np
from PIL import Image, ImageDraw

from ....utils import logging
from ....utils.deps import class_requires_deps, is_dep_available
from ....utils.fonts import SIMFANG_FONT
from ...common.result import (
    BaseCVResult,
    BaseResult,
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
    format_para_title_func,
    format_text_plain_func,
    format_title_func,
    simplify_table_func,
)
from ..ocr.result import draw_box_txt_fine, get_minarea_rect

SKIP_ORDER_LABELS = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
]

if is_dep_available("opencv-contrib-python"):
    import cv2


class PaddleOCRVLBlock(object):
    """PaddleOCRVL Block Class"""

    def __init__(
        self,
        label,
        bbox,
        content="",
        group_id=None,
        polygon_points=None,
        global_block_id=None,
        global_group_id=None,
    ) -> None:
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
        self.polygon_points = polygon_points
        self.group_id = group_id
        self.global_block_id = global_block_id
        self.global_group_id = global_group_id

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
    # get header and rows
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
    # construct html table
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
        "<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    )

    tabel_content = tabel_content.replace(
        "<th>", "<th style='text-align: center; word-wrap: break-word;'>"
    )
    tabel_content = tabel_content.replace(
        "<td>", "<td style='text-align: center; word-wrap: break-word;'>"
    )

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
        "paragraph_title": format_para_title_func,
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
        "spotting": lambda block: block.content,
        "number": format_text_plain_func,
        "footnote": format_text_plain_func,
        "header": format_text_plain_func,
        "header_image": image_func,
        "footer": format_text_plain_func,
        "footer_image": image_func,
        "aside_text": format_text_plain_func,
    }


@class_requires_deps("opencv-contrib-python")
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
        markdown_ignore_labels = self["model_settings"].get(
            "markdown_ignore_labels", []
        )
        self.skip_order_labels = [
            label for label in SKIP_ORDER_LABELS.copy() + markdown_ignore_labels
        ]

    def _to_img(self) -> dict[str, np.ndarray]:
        """
        Convert the parsing result to a dictionary of images.

        Returns:
            dict: Keys are names, values are numpy arrays (images).
        """

        res_img_dict = {}
        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            if isinstance(self["doc_preprocessor_res"], BaseResult):
                for key, value in self["doc_preprocessor_res"].img.items():
                    res_img_dict[key] = value
            if isinstance(self["doc_preprocessor_res"], list):
                for idx, doc_preprocessor_res in enumerate(
                    self["doc_preprocessor_res"]
                ):
                    if isinstance(doc_preprocessor_res, BaseResult):
                        for key, value in doc_preprocessor_res.img.items():
                            res_img_dict[f"{key}_{idx}"] = value
        if self["model_settings"]["use_layout_detection"]:
            if isinstance(self["layout_det_res"], BaseResult):
                res_img_dict["layout_det_res"] = self["layout_det_res"].img["res"]
            if isinstance(self["layout_det_res"], list):
                for idx, layout_res in enumerate(self["layout_det_res"]):
                    if isinstance(layout_res, BaseResult):
                        res_img_dict[f"layout_det_res_{idx}"] = layout_res.img["res"]

        if (
            self.get("spotting_res")
            and not isinstance(self["spotting_res"], list)
            and self.get("doc_preprocessor_res")
        ):
            boxes = self["spotting_res"]["rec_polys"]
            txts = self["spotting_res"]["rec_texts"]
            image = self["doc_preprocessor_res"]["output_img"][:, :, ::-1]
            h, w = image.shape[0:2]
            img_left = Image.fromarray(image)
            img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
            random.seed(0)
            draw_left = ImageDraw.Draw(img_left)
            vis_font = SIMFANG_FONT
            for idx, (box, txt) in enumerate(zip(boxes, txts)):
                try:
                    color = (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    box = np.array(box)
                    if len(box) > 4:
                        pts = [(x, y) for x, y in box.tolist()]
                        draw_left.polygon(pts, outline=color, width=8, fill=color)
                        box = get_minarea_rect(box)
                        height = int(0.5 * (max(box[:, 1]) - min(box[:, 1])))
                        box[:2, 1] = np.mean(box[:, 1])
                        box[2:, 1] = np.mean(box[:, 1]) + min(20, height)
                    else:
                        box_pts = [(int(x), int(y)) for x, y in box.tolist()]
                        draw_left.polygon(box_pts, fill=color)
                    if isinstance(txt, tuple):
                        txt = txt[0]
                    img_right_text = draw_box_txt_fine((w, h), box, txt, vis_font.path)
                    pts = np.array(box, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_right_text, [pts], True, color, 1)
                    img_right = cv2.bitwise_and(img_right, img_right_text)
                except:
                    continue

            img_left = Image.blend(Image.fromarray(image), img_left, 0.5)
            img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
            img_show.paste(img_left, (0, 0, w, h))
            img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))

            res_img_dict["spotting_res_img"] = img_show

        return res_img_dict

    def _to_html(self) -> dict[str, str]:
        """
        Converts the prediction to its corresponding HTML representation.

        Returns:
            dict: The str type HTML representation result.
        """
        res_html_dict = {}
        if self.get("table_res_list") and len(self["table_res_list"]) > 0:
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
        if self.get("table_res_list") and len(self["table_res_list"]) > 0:
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
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        if self["model_settings"]["use_doc_preprocessor"]:
            if isinstance(self["doc_preprocessor_res"], BaseResult):
                data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
            else:
                data["doc_preprocessor_res"] = self["doc_preprocessor_res"]
        if self["model_settings"]["use_layout_detection"]:
            if isinstance(self["layout_det_res"], BaseResult):
                data["layout_det_res"] = self["layout_det_res"].str["res"]
            else:
                data["layout_det_res"] = self["layout_det_res"]
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
        _keep_img = kwargs.pop("keep_img", False)

        data = {}
        data["input_path"] = self["input_path"]
        data["page_index"] = self["page_index"]
        data["page_count"] = self["page_count"]
        data["width"] = self["width"]
        data["height"] = self["height"]
        model_settings = self["model_settings"]
        data["model_settings"] = model_settings
        use_seal_recognition = self["model_settings"].get("use_seal_recognition", False)
        if self["model_settings"].get("format_block_content", False):
            original_image_width = self["width"]
            use_ocr_for_image_block = self["model_settings"].get(
                "use_ocr_for_image_block", False
            )
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_ocr_for_image_block,
                ),
                remove_symbol=not use_ocr_for_image_block,
            )

            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=True,
                ),
                remove_symbol=use_seal_recognition,
            )

            if self["model_settings"].get("use_chart_recognition", False):
                format_chart_func = format_chart2table_func
            else:
                format_chart_func = format_image_func

            if not self["model_settings"].get("use_layout_detection", False):
                format_seal_func = format_text_func

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
            if label not in self.skip_order_labels:
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
                "group_id": (
                    parsing_res.group_id if parsing_res.group_id is not None else idx
                ),
            }
            if (
                hasattr(parsing_res, "global_block_id")
                and parsing_res.global_block_id is not None
            ):
                res_dict["global_block_id"] = parsing_res.global_block_id
            if (
                hasattr(parsing_res, "global_group_id")
                and parsing_res.global_group_id is not None
            ):
                res_dict["global_group_id"] = parsing_res.global_group_id
            if parsing_res.polygon_points is not None:
                res_dict["block_polygon_points"] = parsing_res.polygon_points

            if _keep_img and parsing_res.image is not None:
                res_dict["image"] = parsing_res.image

            if self["model_settings"].get("format_block_content", False):
                if handle_funcs_dict.get(parsing_res.label):
                    res_dict["block_content"] = handle_funcs_dict[parsing_res.label](
                        parsing_res
                    )
                else:
                    res_dict["block_content"] = parsing_res.content

            parsing_res_list_json.append(res_dict)
        data["parsing_res_list"] = parsing_res_list_json
        if self.get("spotting_res"):
            if isinstance(self["spotting_res"], list):
                data["spotting_res"] = [res for res in self["spotting_res"]]
            else:
                data["spotting_res"] = self["spotting_res"]
        if self["model_settings"]["use_doc_preprocessor"]:
            if isinstance(self["doc_preprocessor_res"], BaseResult):
                data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
            elif isinstance(self["doc_preprocessor_res"], list):
                doc_preprocessor_res = []
                for res in self["doc_preprocessor_res"]:
                    if isinstance(res, BaseResult):
                        doc_preprocessor_res.append(res.json["res"])
                    else:
                        doc_preprocessor_res.append(res)
                data["doc_preprocessor_res"] = doc_preprocessor_res
            else:
                data["doc_preprocessor_res"] = self["doc_preprocessor_res"]
        if self["model_settings"]["use_layout_detection"]:
            if isinstance(self["layout_det_res"], BaseResult):
                data["layout_det_res"] = self["layout_det_res"].json["res"]
            elif isinstance(self["layout_det_res"], list):
                layout_det_res = []
                for res in self["layout_det_res"]:
                    if isinstance(res, BaseResult):
                        layout_det_res.append(res.json["res"])
                    else:
                        layout_det_res.append(res)
                data["layout_det_res"] = layout_det_res
            else:
                data["layout_det_res"] = self["layout_det_res"]

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

        use_ocr_for_image_block = self["model_settings"].get(
            "use_ocr_for_image_block", False
        )
        use_seal_recognition = self["model_settings"].get("use_seal_recognition", False)
        if isinstance(self["width"], list):
            original_image_width = self["width"][0]
        else:
            original_image_width = self["width"]

        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_ocr_for_image_block,
                ),
                remove_symbol=not use_ocr_for_image_block,
            )
            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_seal_recognition,
                ),
                remove_symbol=False,
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = lambda block: format_image_plain_func(
                block, use_ocr_for_image_block
            )
            format_seal_func = lambda block: format_image_plain_func(
                block, use_seal_recognition
            )

        format_chart_func = (
            format_chart2table_func
            if self["model_settings"]["use_chart_recognition"]
            else format_image_func
        )

        if not self["model_settings"].get("use_layout_detection", False):
            format_seal_func = format_text_func

        if pretty:
            format_table_func = lambda block: "\n" + format_table_center_func(block)
        else:
            format_table_func = lambda block: simplify_table_func("\n" + block.content)

        format_formula_func = lambda block: block.content

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )
        for label in self["model_settings"].get("markdown_ignore_labels", []):
            handle_funcs_dict.pop(label, None)

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


class PaddleOCRVLPagesResult(PaddleOCRVLResult):
    def save_to_img(self, *args, **kwargs):
        logging.warning(
            f"The result of multi-pages don't support to save as image format!"
        )
        return None

    def save_to_html(self, *args, **kwargs):
        logging.warning(
            f"The result of multi-pages don't support to save as html format!"
        )
        return None

    def save_to_xlsx(self, *args, **kwargs):
        logging.warning(
            f"The result of multi-pages don't support to save as xlsx format!"
        )
        return None

    def _to_markdown(self, pretty=True, show_formula_number=False) -> dict:
        """
        Save the parsing result to a Markdown file.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.
            show_formula_number (bool): whether to show formula numbers.

        Returns:
            dict: Markdown information with text and images.
        """

        use_ocr_for_image_block = self["model_settings"].get(
            "use_ocr_for_image_block", False
        )
        use_seal_recognition = self["model_settings"].get("use_seal_recognition", False)
        if isinstance(self["width"], list):
            original_image_width = self["width"][0]
        else:
            original_image_width = self["width"]

        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_ocr_for_image_block,
                ),
                remove_symbol=not use_ocr_for_image_block,
            )
            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_seal_recognition,
                ),
                remove_symbol=False,
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = lambda block: format_image_plain_func(
                block, use_ocr_for_image_block
            )
            format_seal_func = lambda block: format_image_plain_func(
                block, use_seal_recognition
            )

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

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )
        for label in self["model_settings"].get("markdown_ignore_labels", []):
            handle_funcs_dict.pop(label, None)

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
