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

import os
import re
import json
import numpy as np
from .utils import *
from copy import deepcopy
from ...components import *
from ..ocr import OCRPipeline
from ....utils import logging
from ...results import (
    TableResult,
    LayoutStructureResult,
    VisualInfoResult,
    ChatOCRResult,
)
from ...components.llm import ErnieBot
from ...utils.io import ImageReader, PDFReader
from ..table_recognition import TableRecPipeline
from ...components.llm import create_llm_api, ErnieBot
from ....utils.file_interface import read_yaml_file
from ..table_recognition.utils import convert_4point2rect, get_ori_coordinate_for_table

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "ch_prompt.yaml")


class PPChatOCRPipeline(TableRecPipeline):
    """PP-ChatOCRv3 Pileline"""

    entities = "chatocrv3"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        oricls_model=None,
        uvdoc_model=None,
        curve_model=None,
        llm_name="ernie-3.5",
        llm_params={},
        task_prompt_yaml=None,
        user_prompt_yaml=None,
        layout_batch_size=1,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        table_batch_size=1,
        uvdoc_batch_size=1,
        curve_batch_size=1,
        oricls_batch_size=1,
        recovery=True,
        device="gpu",
        predictor_kwargs=None,
    ):
        self.layout_model = layout_model
        self.text_det_model = text_det_model
        self.text_rec_model = text_rec_model
        self.table_model = table_model
        self.oricls_model = oricls_model
        self.uvdoc_model = uvdoc_model
        self.curve_model = curve_model
        self.llm_name = llm_name
        self.llm_params = llm_params
        self.task_prompt_yaml = task_prompt_yaml
        self.user_prompt_yaml = user_prompt_yaml
        self.layout_batch_size = layout_batch_size
        self.text_det_batch_size = text_det_batch_size
        self.text_rec_batch_size = text_rec_batch_size
        self.table_batch_size = table_batch_size
        self.uvdoc_batch_size = uvdoc_batch_size
        self.curve_batch_size = curve_batch_size
        self.oricls_batch_size = oricls_batch_size
        self.recovery = recovery
        self.device = device
        self.predictor_kwargs = predictor_kwargs
        super().__init__(
            layout_model=layout_model,
            text_det_model=text_det_model,
            text_rec_model=text_rec_model,
            table_model=table_model,
            layout_batch_size=layout_batch_size,
            text_det_batch_size=text_det_batch_size,
            text_rec_batch_size=text_rec_batch_size,
            table_batch_size=table_batch_size,
            predictor_kwargs=predictor_kwargs,
        )
        self._build_predictor()
        self.llm_api = create_llm_api(
            llm_name,
            llm_params,
        )
        self.cropper = CropByBoxes()
        # get base prompt from yaml info
        if task_prompt_yaml:
            self.task_prompt_dict = read_yaml_file(task_prompt_yaml)
        else:
            self.task_prompt_dict = read_yaml_file(
                PROMPT_FILE
            )  # get user prompt from yaml info
        if user_prompt_yaml:
            self.user_prompt_dict = read_yaml_file(user_prompt_yaml)
        else:
            self.user_prompt_dict = None
        self.recovery = recovery
        self.img_reader = ReadImage()
        self.pdf_reader = PDFReader()
        self.visual_info = None
        self.vector = None
        self._set_predictor(
            oricls_batch_size, uvdoc_batch_size, curve_batch_size, device=device
        )

    def _build_predictor(self):
        super()._build_predictor()
        if self.curve_model:
            self.curve_pipeline = OCRPipeline(
                text_det_model=self.curve_model,
                text_rec_model=self.text_rec_model,
                text_det_batch_size=self.text_det_batch_size,
                text_rec_batch_size=self.text_rec_batch_size,
                predictor_kwargs=self.predictor_kwargs,
            )
        else:
            self.curve_pipeline = None
        if self.oricls_model:
            self.oricls_predictor = self._create_model(self.oricls_model)
        else:
            self.oricls_predictor = None
        if self.uvdoc_model:
            self.uvdoc_predictor = self._create_model(self.uvdoc_model)
        else:
            self.uvdoc_predictor = None
        if self.curve_pipeline and self.curve_batch_size:
            self.curve_pipeline.text_det_model.set_predictor(
                batch_size=self.curve_batch_size, device=self.device
            )
        if self.oricls_predictor and self.oricls_batch_size:
            self.oricls_predictor.set_predictor(
                batch_size=self.oricls_batch_size, device=self.device
            )
        if self.uvdoc_predictor and self.uvdoc_batch_size:
            self.uvdoc_predictor.set_predictor(
                batch_size=self.uvdoc_batch_size, device=self.device
            )

    def _set_predictor(
        self, curve_batch_size, oricls_batch_size, uvdoc_batch_size, device
    ):
        if self.curve_pipeline and curve_batch_size:
            self.curve_pipeline.text_det_model.set_predictor(
                batch_size=curve_batch_size, device=device
            )
        if self.oricls_predictor and oricls_batch_size:
            self.oricls_predictor.set_predictor(
                batch_size=oricls_batch_size, device=device
            )
        if self.uvdoc_predictor and uvdoc_batch_size:
            self.uvdoc_predictor.set_predictor(
                batch_size=uvdoc_batch_size, device=device
            )

    def predict(self, input, **kwargs):
        visual_info = {"ocr_text": [], "table_html": [], "table_text": []}
        # get all visual result
        visual_result = list(self.get_visual_result(input, **kwargs))
        # decode visual result to get table_html, table_text, ocr_text
        ocr_text, table_text, table_html = self.decode_visual_result(visual_result)

        visual_info["ocr_text"] = ocr_text
        visual_info["table_html"] = table_html
        visual_info["table_text"] = table_text
        visual_info = VisualInfoResult(visual_info)
        # for local user save visual info in self
        self.visual_info = visual_info

        return visual_result, visual_info

    def get_visual_result(self, inputs, **kwargs):
        curve_batch_size = kwargs.get("curve_batch_size")
        oricls_batch_size = kwargs.get("oricls_batch_size")
        uvdoc_batch_size = kwargs.get("uvdoc_batch_size")
        device = kwargs.get("device")
        self._set_predictor(
            curve_batch_size, oricls_batch_size, uvdoc_batch_size, device
        )
        input_imgs = []
        img_list = []
        for file in inputs:
            if isinstance(file, str) and file.endswith(".pdf"):
                img_list = self.pdf_reader.read(file)
                for page, img in enumerate(img_list):
                    input_imgs.append(
                        {
                            "input_path": f"{Path(file).parent}/{Path(file).stem}_{page}.jpg",
                            "img": img,
                        }
                    )
            else:
                for imgs in self.img_reader(file):
                    input_imgs.extend(imgs)
        # get oricls and uvdoc results
        oricls_results = []
        if self.oricls_predictor and kwargs.get("use_oricls_model", True):
            img_list = [img["img"] for img in input_imgs]
            oricls_results = get_oriclas_results(
                input_imgs, self.oricls_predictor, img_list
            )
        uvdoc_results = []
        if self.uvdoc_predictor and kwargs.get("use_uvdoc_model", True):
            img_list = [img["img"] for img in input_imgs]
            uvdoc_results = get_uvdoc_results(
                input_imgs, self.uvdoc_predictor, img_list
            )
        img_list = [img["img"] for img in input_imgs]
        for idx, (input_img, layout_pred) in enumerate(
            zip(input_imgs, self.layout_predictor(img_list))
        ):
            single_img_res = {
                "input_path": "",
                "layout_result": {},
                "ocr_result": {},
                "table_ocr_result": [],
                "table_result": [],
                "structure_result": [],
                "structure_result": [],
                "oricls_result": {},
                "uvdoc_result": {},
                "curve_result": [],
            }
            # update oricls and uvdoc result
            if oricls_results:
                single_img_res["oricls_result"] = oricls_results[idx]
            if uvdoc_results:
                single_img_res["uvdoc_result"] = uvdoc_results[idx]
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred
            single_img = input_img["img"]
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                # get cropped images with label "table"
                table_subs = []
                curve_subs = []
                structure_res = []
                ocr_res_with_layout = []
                for sub in subs_of_img:
                    box = sub["box"]
                    xmin, ymin, xmax, ymax = [int(i) for i in box]
                    mask_flag = True
                    if sub["label"].lower() == "table":
                        table_subs.append(sub)
                    elif sub["label"].lower() == "seal":
                        curve_subs.append(sub)
                    else:
                        if self.recovery and kwargs.get("recovery", True):
                            # TODO: Why use the entire image?
                            wht_im = (
                                np.ones(single_img.shape, dtype=single_img.dtype) * 255
                            )
                            wht_im[ymin:ymax, xmin:xmax, :] = sub["img"]
                            sub_ocr_res = get_ocr_res(self.ocr_pipeline, wht_im)
                        else:
                            sub_ocr_res = get_ocr_res(self.ocr_pipeline, sub)
                            sub_ocr_res["dt_polys"] = get_ori_coordinate_for_table(
                                xmin, ymin, sub_ocr_res["dt_polys"]
                            )
                        layout_label = sub["label"].lower()
                        if sub_ocr_res and sub["label"].lower() in [
                            "image",
                            "figure",
                            "img",
                            "fig",
                        ]:
                            mask_flag = False
                        else:
                            ocr_res_with_layout.append(sub_ocr_res)
                            structure_res.append(
                                {
                                    "layout_bbox": box,
                                    f"{layout_label}": "\n".join(
                                        sub_ocr_res["rec_text"]
                                    ),
                                }
                            )
                    if mask_flag:
                        single_img[ymin:ymax, xmin:xmax, :] = 255

            curve_pipeline = self.ocr_pipeline
            if self.curve_pipeline and kwargs.get("use_curve_model", True):
                curve_pipeline = self.curve_pipeline

            all_curve_res = get_ocr_res(curve_pipeline, curve_subs)
            single_img_res["curve_result"] = all_curve_res

            for sub, curve_res in zip(curve_subs, all_curve_res):
                structure_res.append(
                    {
                        "layout_bbox": sub["box"],
                        "印章": "".join(curve_res["rec_text"]),
                    }
                )

            ocr_res = get_ocr_res(self.ocr_pipeline, single_img)
            ocr_res["input_path"] = layout_pred["input_path"]
            all_table_res, _ = self.get_table_result(table_subs)
            for idx, single_dt_poly in enumerate(ocr_res["dt_polys"]):
                structure_res.append(
                    {
                        "layout_bbox": convert_4point2rect(single_dt_poly),
                        "words in text block": ocr_res["rec_text"][idx],
                    }
                )
            # update ocr result
            for layout_ocr_res in ocr_res_with_layout:
                ocr_res["dt_polys"].extend(layout_ocr_res["dt_polys"])
                ocr_res["rec_text"].extend(layout_ocr_res["rec_text"])
                ocr_res["input_path"] = single_img_res["input_path"]

            all_table_ocr_res = []
            # get table text from html
            structure_res_table, all_table_ocr_res = get_table_text_from_html(
                all_table_res
            )
            structure_res.extend(structure_res_table)

            # sort the layout result by the left top point of the box
            structure_res = sorted_layout_boxes(structure_res, w=single_img.shape[1])
            structure_res = [LayoutStructureResult(item) for item in structure_res]

            single_img_res["table_result"] = all_table_res
            single_img_res["ocr_result"] = ocr_res
            single_img_res["table_ocr_result"] = all_table_ocr_res
            single_img_res["structure_result"] = structure_res

            yield ChatOCRResult(single_img_res)

    def decode_visual_result(self, visual_result):
        ocr_text = []
        table_text_list = []
        table_html = []
        for single_img_pred in visual_result:
            layout_res = single_img_pred["structure_result"]
            layout_res_copy = deepcopy(layout_res)
            # layout_res is [{"layout_bbox": [x1, y1, x2, y2], "layout": "single","words in text block":"xxx"}, {"layout_bbox": [x1, y1, x2, y2], "layout": "double","印章":"xxx"}
            ocr_res = {}
            for block in layout_res_copy:
                block.pop("layout_bbox")
                block.pop("layout")
                for layout_type, text in block.items():
                    if text == "":
                        continue
                    # Table results are used separately
                    if layout_type == "table":
                        continue
                    if layout_type not in ocr_res:
                        ocr_res[layout_type] = text
                    else:
                        ocr_res[layout_type] += f"\n {text}"

            single_table_text = " ".join(single_img_pred["table_ocr_result"])
            for table_pred in single_img_pred["table_result"]:
                html = table_pred["html"]
                table_html.append(html)
            if ocr_res:
                ocr_text.append(ocr_res)
            table_text_list.append(single_table_text)

        return ocr_text, table_text_list, table_html

    def get_vector_text(
        self,
        llm_name=None,
        llm_params={},
        visual_info=None,
        min_characters=0,
        llm_request_interval=1.0,
    ):
        """get vector for ocr"""
        if isinstance(self.llm_api, ErnieBot):
            get_vector_flag = True
        else:
            logging.warning("Do not use ErnieBot, will not get vector text.")
            get_vector_flag = False
        if not any([visual_info, self.visual_info]):
            return {"vector": None}

        if visual_info:
            # use for serving or local
            _visual_info = visual_info
        else:
            # use for local
            _visual_info = self.visual_info

        ocr_text = _visual_info["ocr_text"]
        html_list = _visual_info["table_html"]
        table_text_list = _visual_info["table_text"]

        # add table text to ocr text
        for html, table_text_rec in zip(html_list, table_text_list):
            if len(html) > 3000:
                ocr_text.append({"table": table_text_rec})

        ocr_all_result = "".join(["\n".join(e.values()) for e in ocr_text])

        if len(ocr_all_result) > min_characters and get_vector_flag:
            if visual_info and llm_name:
                # for serving or local
                llm_api = create_llm_api(llm_name, llm_params)
                text_result = llm_api.get_vector(ocr_text, llm_request_interval)
            else:
                # for local
                text_result = self.llm_api.get_vector(ocr_text, llm_request_interval)
        else:
            text_result = str(ocr_text)

        return {"vector": text_result}

    def get_retrieval_text(
        self,
        key_list,
        visual_info=None,
        vector=None,
        llm_name=None,
        llm_params={},
        llm_request_interval=0.1,
    ):

        if not any([visual_info, vector, self.visual_info, self.vector]):
            return {"retrieval": None}

        key_list = format_key(key_list)

        if not any([vector, self.vector]):
            logging.warning(
                "The vector library is not created, and is being created automatically"
            )
            if visual_info and llm_name:
                # for serving
                vector = self.get_vector_text(
                    llm_name=llm_name, llm_params=llm_params, visual_info=visual_info
                )
            else:
                self.vector = self.get_vector_text()

        if vector and llm_name:
            _vector = vector["vector"]
            llm_api = create_llm_api(llm_name, llm_params)
            retrieval = llm_api.caculate_similar(
                vector=_vector,
                key_list=key_list,
                llm_params=llm_params,
                sleep_time=llm_request_interval,
            )
        else:
            _vector = self.vector["vector"]
            retrieval = self.llm_api.caculate_similar(
                vector=_vector, key_list=key_list, sleep_time=llm_request_interval
            )

        return {"retrieval": retrieval}

    def chat(
        self,
        key_list,
        vector=None,
        visual_info=None,
        retrieval_result=None,
        user_task_description="",
        rules="",
        few_shot="",
        use_vector=True,
        save_prompt=False,
        llm_name="ernie-3.5",
        llm_params={},
    ):
        """
        chat with key

        """
        if not any(
            [vector, visual_info, retrieval_result, self.visual_info, self.vector]
        ):
            return {"chat_res": "请先完成图像解析再开始再对话", "prompt": ""}
        key_list = format_key(key_list)
        # first get from table, then get from text in table, last get from all ocr
        if visual_info:
            # use for serving or local
            _visual_info = visual_info
        else:
            # use for local
            _visual_info = self.visual_info

        ocr_text = _visual_info["ocr_text"]
        html_list = _visual_info["table_html"]
        table_text_list = _visual_info["table_text"]
        if retrieval_result:
            ocr_text = retrieval_result
        elif use_vector and any([visual_info, vector]):
            # for serving or local
            ocr_text = self.get_retrieval_text(
                key_list=key_list,
                visual_info=visual_info,
                vector=vector,
                llm_name=llm_name,
                llm_params=llm_params,
            )
        else:
            # for local
            ocr_text = self.get_retrieval_text(key_list=key_list)

        prompt_res = {"ocr_prompt": "str", "table_prompt": [], "html_prompt": []}

        final_results = {}
        failed_results = ["大模型调用失败", "未知", "未找到关键信息", "None", ""]
        if html_list:
            prompt_list = self.get_prompt_for_table(
                html_list, key_list, rules, few_shot
            )
            prompt_res["html_prompt"] = prompt_list
            for prompt, table_text in zip(prompt_list, table_text_list):
                logging.debug(prompt)
                res = self.get_llm_result(prompt)
                # TODO: why use one html but the whole table_text in next step
                if list(res.values())[0] in failed_results:
                    logging.info(
                        "table html sequence is too much longer, using ocr directly"
                    )
                    prompt = self.get_prompt_for_ocr(
                        table_text, key_list, rules, few_shot, user_task_description
                    )
                    logging.debug(prompt)
                    prompt_res["table_prompt"].append(prompt)
                    res = self.get_llm_result(prompt)
                for key, value in res.items():
                    if value not in failed_results and key in key_list:
                        key_list.remove(key)
                        final_results[key] = value
        if len(key_list) > 0:
            logging.info("get result from ocr")
            prompt = self.get_prompt_for_ocr(
                ocr_text,
                key_list,
                rules,
                few_shot,
                user_task_description,
            )
            logging.debug(prompt)
            prompt_res["ocr_prompt"] = prompt
            res = self.get_llm_result(prompt)
            final_results.update(res)
        if not res and not final_results:
            final_results = self.llm_api.ERROR_MASSAGE
        if save_prompt:
            return {"chat_res": final_results, "prompt": prompt_res}
        else:
            return {"chat_res": final_results, "prompt": ""}

    def get_llm_result(self, prompt):
        """get llm result and decode to dict"""
        llm_result = self.llm_api.pred(prompt)
        # when the llm pred failed, return None
        if not llm_result:
            return None

        if "json" in llm_result or "```" in llm_result:
            llm_result = (
                llm_result.replace("```", "").replace("json", "").replace("/n", "")
            )

            llm_result = llm_result.replace("[", "").replace("]", "")
        try:
            llm_result = json.loads(llm_result)
            llm_result_final = {}
            for key in llm_result:
                value = llm_result[key]
                if isinstance(value, list):
                    if len(value) > 0:
                        llm_result_final[key] = value[0]
                else:
                    llm_result_final[key] = value
            return llm_result_final
        except:
            results = (
                llm_result.replace("\n", "")
                .replace("    ", "")
                .replace("{", "")
                .replace("}", "")
            )
            if not results.endswith('"'):
                results = results + '"'
            pattern = r'"(.*?)": "([^"]*)"'
            matches = re.findall(pattern, str(results))
            llm_result = {k: v for k, v in matches}
            return llm_result

    def get_prompt_for_table(self, table_result, key_list, rules="", few_shot=""):
        """get prompt for table"""
        prompt_key_information = []
        merge_table = ""
        for idx, result in enumerate(table_result):
            if len(merge_table + result) < 2000:
                merge_table += result
            if len(merge_table + result) > 2000 or idx == len(table_result) - 1:
                single_prompt = self.get_kie_prompt(
                    merge_table,
                    key_list,
                    rules_str=rules,
                    few_shot_demo_str=few_shot,
                    prompt_type="table",
                )
                prompt_key_information.append(single_prompt)
                merge_table = ""
        return prompt_key_information

    def get_prompt_for_ocr(
        self,
        ocr_result,
        key_list,
        rules="",
        few_shot="",
        user_task_description="",
    ):
        """get prompt for ocr"""

        prompt_key_information = self.get_kie_prompt(
            ocr_result, key_list, user_task_description, rules, few_shot
        )
        return prompt_key_information

    def get_kie_prompt(
        self,
        text_result,
        key_list,
        user_task_description="",
        rules_str="",
        few_shot_demo_str="",
        prompt_type="common",
    ):
        """get_kie_prompt"""

        if prompt_type == "table":
            task_description = self.task_prompt_dict["kie_table_prompt"][
                "task_description"
            ]
        else:
            task_description = self.task_prompt_dict["kie_common_prompt"][
                "task_description"
            ]
            output_format = self.task_prompt_dict["kie_common_prompt"]["output_format"]
            if len(user_task_description) > 0:
                task_description = user_task_description
            task_description = task_description + output_format

        few_shot_demo_key_value = ""

        if self.user_prompt_dict:
            logging.info("======= common use custom ========")
            task_description = self.user_prompt_dict["task_description"]
            rules_str = self.user_prompt_dict["rules_str"]
            few_shot_demo_str = self.user_prompt_dict["few_shot_demo_str"]
            few_shot_demo_key_value = self.user_prompt_dict["few_shot_demo_key_value"]

        prompt = f"""{task_description}{rules_str}{few_shot_demo_str}{few_shot_demo_key_value}"""

        if prompt_type == "table":
            prompt += f"""\n结合上面，下面正式开始：\
                表格内容：```{text_result}```\
                关键词列表：[{key_list}]。""".replace(
                "    ", ""
            )
        else:
            prompt += f"""\n结合上面的例子，下面正式开始：\
                OCR文字：```{text_result}```\
                关键词列表：[{key_list}]。""".replace(
                "    ", ""
            )

        return prompt
