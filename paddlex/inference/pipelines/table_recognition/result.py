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

import copy
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw

from ...common.result import BaseCVResult, HtmlMixin, JsonMixin, XlsxMixin


class SingleTableRecognitionResult(BaseCVResult, HtmlMixin, XlsxMixin):
    """single table recognition result"""

    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self["page_index"]) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            return f"{stem}_{page_idx}{suffix}"
        else:
            return fn

    def _to_html(self) -> Dict[str, str]:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        return {"pred": self["pred_html"]}

    def _to_xlsx(self) -> Dict[str, str]:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            str: The path to the XLSX file containing the prediction data.
        """
        return {"pred": self["pred_html"]}

    def _to_str(self, *args, **kwargs) -> Dict[str, str]:
        """Converts the instance's attributes to a dictionary and then to a string.

        Args:
            *args: Additional positional arguments passed to the base class method.
            **kwargs: Additional keyword arguments passed to the base class method.

        Returns:
            Dict[str, str]: A dictionary with the instance's attributes converted to strings.
        """
        data = {}
        data["cell_box_list"] = self["cell_box_list"]
        data["pred_html"] = self["pred_html"]
        data["table_ocr_pred"] = self["table_ocr_pred"]
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
        data["cell_box_list"] = self["cell_box_list"]
        data["pred_html"] = self["pred_html"]
        data["table_ocr_pred"] = self["table_ocr_pred"]
        return JsonMixin._to_json(data, *args, **kwargs)


class TableRecognitionResult(BaseCVResult, HtmlMixin, XlsxMixin):
    """Table Recognition Result"""

    def __init__(self, data: Dict) -> None:
        super().__init__(data)
        HtmlMixin.__init__(self)
        XlsxMixin.__init__(self)

    def _get_input_fn(self):
        fn = super()._get_input_fn()
        if (page_idx := self["page_index"]) is not None:
            fp = Path(fn)
            stem, suffix = fp.stem, fp.suffix
            return f"{stem}_{page_idx}{suffix}"
        else:
            return fn

    def _to_img(self) -> Dict[str, np.ndarray]:
        res_img_dict = {}
        layout_det_res = self["layout_det_res"]
        if len(layout_det_res) > 0:
            res_img_dict["layout_det_res"] = layout_det_res.img["res"]

        model_settings = self["model_settings"]
        if model_settings["use_doc_preprocessor"]:
            res_img_dict.update(**self["doc_preprocessor_res"].img)

        res_img_dict.update(**self["overall_ocr_res"].img)

        if len(self["table_res_list"]) > 0:
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
        return res_img_dict

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
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].str["res"]
        if len(self["layout_det_res"]) > 0:
            data["layout_det_res"] = self["layout_det_res"].str["res"]
        data["overall_ocr_res"] = self["overall_ocr_res"].str["res"]
        data["table_res_list"] = []
        for sno in range(len(self["table_res_list"])):
            table_res = self["table_res_list"][sno]
            data["table_res_list"].append(table_res.str["res"])
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
        if self["model_settings"]["use_doc_preprocessor"]:
            data["doc_preprocessor_res"] = self["doc_preprocessor_res"].json["res"]
        if len(self["layout_det_res"]) > 0:
            data["layout_det_res"] = self["layout_det_res"].json["res"]
        data["overall_ocr_res"] = self["overall_ocr_res"].json["res"]
        data["table_res_list"] = []
        for sno in range(len(self["table_res_list"])):
            table_res = self["table_res_list"][sno]
            data["table_res_list"].append(table_res.json["res"])
        return JsonMixin._to_json(data, *args, **kwargs)

    def _to_html(self) -> Dict[str, str]:
        """Converts the prediction to its corresponding HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        res_html_dict = {}
        for sno in range(len(self["table_res_list"])):
            table_res = self["table_res_list"][sno]
            table_region_id = table_res["table_region_id"]
            key = f"table_{table_region_id}"
            res_html_dict[key] = table_res.html["pred"]
            res_html_dict[key] = res_html_dict[key].replace(
                "<table>", '<table border="1">'
            )
        return res_html_dict

    def _to_xlsx(self) -> Dict[str, str]:
        """Converts the prediction HTML to an XLSX file path.

        Returns:
            Dict[str, str]: The str type XLSX representation result.
        """
        res_xlsx_dict = {}
        for sno in range(len(self["table_res_list"])):
            table_res = self["table_res_list"][sno]
            table_region_id = table_res["table_region_id"]
            key = f"table_{table_region_id}"
            res_xlsx_dict[key] = table_res.xlsx["pred"]
        return res_xlsx_dict
