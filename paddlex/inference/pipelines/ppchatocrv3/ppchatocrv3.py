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
from ...results import *
from copy import deepcopy
from ...components import *
from ..ocr import OCRPipeline
from ....utils import logging
from ...components.llm import ErnieBot
from ..table_recognition import _TableRecPipeline
from ...components.llm import create_llm_api, ErnieBot
from ....utils.file_interface import read_yaml_file
from ..table_recognition.utils import convert_4point2rect, get_ori_coordinate_for_table

PROMPT_FILE = os.path.join(os.path.dirname(__file__), "ch_prompt.yaml")


class PPChatOCRPipeline(_TableRecPipeline):
    """PP-ChatOCRv3 Pileline"""

    entities = "PP-ChatOCRv3-doc"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        doc_image_ori_cls_model=None,
        doc_image_unwarp_model=None,
        seal_text_det_model=None,
        llm_name="ernie-3.5",
        llm_params={},
        task_prompt_yaml=None,
        user_prompt_yaml=None,
        layout_batch_size=1,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        table_batch_size=1,
        doc_image_ori_cls_batch_size=1,
        doc_image_unwarp_batch_size=1,
        seal_text_det_batch_size=1,
        recovery=True,
        device=None,
        predictor_kwargs=None,
        _build_models=True,
    ):
        super().__init__(device, predictor_kwargs)
        if _build_models:
            self._build_predictor(
                layout_model=layout_model,
                text_det_model=text_det_model,
                text_rec_model=text_rec_model,
                table_model=table_model,
                doc_image_ori_cls_model=doc_image_ori_cls_model,
                doc_image_unwarp_model=doc_image_unwarp_model,
                seal_text_det_model=seal_text_det_model,
                llm_name=llm_name,
                llm_params=llm_params,
            )
            self.set_predictor(
                layout_batch_size=layout_batch_size,
                text_det_batch_size=text_det_batch_size,
                text_rec_batch_size=text_rec_batch_size,
                table_batch_size=table_batch_size,
                doc_image_ori_cls_batch_size=doc_image_ori_cls_batch_size,
                doc_image_unwarp_batch_size=doc_image_unwarp_batch_size,
                seal_text_det_batch_size=seal_text_det_batch_size,
            )
        else:
            self.llm_api = create_llm_api(
                llm_name,
                llm_params,
            )

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
        self.visual_info = None
        self.vector = None
        self.visual_flag = False

    def _build_predictor(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        llm_name,
        llm_params,
        seal_text_det_model=None,
        doc_image_ori_cls_model=None,
        doc_image_unwarp_model=None,
    ):
        super()._build_predictor(
            layout_model, text_det_model, text_rec_model, table_model
        )
        if seal_text_det_model:
            self.curve_pipeline = self._create(
                pipeline=OCRPipeline,
                text_det_model=seal_text_det_model,
                text_rec_model=text_rec_model,
            )
        else:
            self.curve_pipeline = None
        if doc_image_ori_cls_model:
            self.doc_image_ori_cls_predictor = self._create(doc_image_ori_cls_model)
        else:
            self.doc_image_ori_cls_predictor = None
        if doc_image_unwarp_model:
            self.doc_image_unwarp_predictor = self._create(doc_image_unwarp_model)
        else:
            self.doc_image_unwarp_predictor = None

        self.img_reader = ReadImage(format="BGR")
        self.llm_api = create_llm_api(
            llm_name,
            llm_params,
        )
        self.cropper = CropByBoxes()

    def set_predictor(
        self,
        layout_batch_size=None,
        text_det_batch_size=None,
        text_rec_batch_size=None,
        table_batch_size=None,
        doc_image_ori_cls_batch_size=None,
        doc_image_unwarp_batch_size=None,
        seal_text_det_batch_size=None,
        device=None,
    ):
        if text_det_batch_size and text_det_batch_size > 1:
            logging.warning(
                f"text det model only support batch_size=1 now,the setting of text_det_batch_size={text_det_batch_size} will not using! "
            )
        if layout_batch_size:
            self.layout_predictor.set_predictor(batch_size=layout_batch_size)
        if text_rec_batch_size:
            self.ocr_pipeline.text_rec_model.set_predictor(
                batch_size=text_rec_batch_size
            )
        if table_batch_size:
            self.table_predictor.set_predictor(batch_size=table_batch_size)
        if self.curve_pipeline and seal_text_det_batch_size:
            self.curve_pipeline.text_det_model.set_predictor(
                batch_size=seal_text_det_batch_size
            )
        if self.doc_image_ori_cls_predictor and doc_image_ori_cls_batch_size:
            self.doc_image_ori_cls_predictor.set_predictor(
                batch_size=doc_image_ori_cls_batch_size
            )
        if self.doc_image_unwarp_predictor and doc_image_unwarp_batch_size:
            self.doc_image_unwarp_predictor.set_predictor(
                batch_size=doc_image_unwarp_batch_size
            )

        if device:
            if self.curve_pipeline:
                self.curve_pipeline.set_predictor(device=device)
            if self.doc_image_ori_cls_predictor:
                self.doc_image_ori_cls_predictor.set_predictor(device=device)
            if self.doc_image_unwarp_predictor:
                self.doc_image_unwarp_predictor.set_predictor(device=device)
            self.layout_predictor.set_predictor(device=device)
            self.ocr_pipeline.set_predictor(device=device)

    def predict(self, *args, **kwargs):
        logging.error(
            "PP-ChatOCRv3-doc Pipeline do not support to call `predict()` directly! Please call `visual_predict(input)` firstly to get visual prediction of `input` and call `chat(key_list)` to get the result of query specified by `key_list`."
        )
        return

    def visual_predict(
        self,
        input,
        use_doc_image_ori_cls_model=True,
        use_doc_image_unwarp_model=True,
        use_seal_text_det_model=True,
        recovery=True,
        **kwargs,
    ):
        self.set_predictor(**kwargs)

        visual_info = {"ocr_text": [], "table_html": [], "table_text": []}
        # get all visual result
        visual_result = list(
            self.get_visual_result(
                input,
                use_doc_image_ori_cls_model=use_doc_image_ori_cls_model,
                use_doc_image_unwarp_model=use_doc_image_unwarp_model,
                use_seal_text_det_model=use_seal_text_det_model,
                recovery=recovery,
            )
        )
        # decode visual result to get table_html, table_text, ocr_text
        ocr_text, table_text, table_html = self.decode_visual_result(visual_result)

        visual_info["ocr_text"] = ocr_text
        visual_info["table_html"] = table_html
        visual_info["table_text"] = table_text
        visual_info = VisualInfoResult(visual_info)
        # for local user save visual info in self
        self.visual_info = visual_info
        self.visual_flag = True

        return visual_result, visual_info

    def get_visual_result(
        self,
        inputs,
        use_doc_image_ori_cls_model=True,
        use_doc_image_unwarp_model=True,
        use_seal_text_det_model=True,
        recovery=True,
    ):
        # get oricls and unwarp results
        if isinstance(inputs, str):
            img_info_list = list(self.img_reader(inputs))[0]
        elif isinstance(inputs, list):
            assert not any(
                isinstance(s, str) and s.endswith(".pdf") for s in inputs
            ), "List containing pdf is not supported; only a list of images or a single PDF is supported."
            img_info_list = [x[0] for x in list(self.img_reader(inputs))]

        oricls_results = []
        if self.doc_image_ori_cls_predictor and use_doc_image_ori_cls_model:
            oricls_results = get_oriclas_results(
                img_info_list, self.doc_image_ori_cls_predictor
            )
        unwarp_results = []
        if self.doc_image_unwarp_predictor and use_doc_image_unwarp_model:
            unwarp_results = get_unwarp_results(
                img_info_list, self.doc_image_unwarp_predictor
            )
        img_list = [img_info["img"] for img_info in img_info_list]

        for idx, (img_info, layout_pred) in enumerate(
            zip(img_info_list, self.layout_predictor(img_list))
        ):
            page_id = idx
            single_img_res = {
                "input_path": "",
                "layout_result": DetResult({}),
                "ocr_result": OCRResult({}),
                "table_ocr_result": [],
                "table_result": StructureTableResult([]),
                "layout_parsing_result": {},
                "oricls_result": TopkResult({}),
                "unwarp_result": DocTrResult({}),
                "curve_result": [],
            }
            # update oricls and unwarp results
            if oricls_results:
                single_img_res["oricls_result"] = oricls_results[idx]
            if unwarp_results:
                single_img_res["unwarp_result"] = unwarp_results[idx]
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred
            single_img = img_info["img"]
            table_subs = []
            curve_subs = []
            structure_res = []
            ocr_res_with_layout = []
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                # get cropped images
                for sub in subs_of_img:
                    box = sub["box"]
                    xmin, ymin, xmax, ymax = [int(i) for i in box]
                    mask_flag = True
                    if sub["label"].lower() == "table":
                        table_subs.append(sub)
                    elif sub["label"].lower() == "seal":
                        curve_subs.append(sub)
                    else:
                        if self.recovery and recovery:
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
            if self.curve_pipeline and use_seal_text_det_model:
                curve_pipeline = self.curve_pipeline

            all_curve_res = get_ocr_res(curve_pipeline, curve_subs)
            single_img_res["curve_result"] = all_curve_res
            if isinstance(all_curve_res, dict):
                all_curve_res = [all_curve_res]
            for sub, curve_res in zip(curve_subs, all_curve_res):
                dt_polys_list = [
                    list(map(list, sublist)) for sublist in curve_res["dt_polys"]
                ]
                sorted_items = sorted(
                    zip(dt_polys_list, curve_res["rec_text"]),
                    key=lambda x: (x[0][0][1], x[0][0][0]),
                )
                _, sorted_text = zip(*sorted_items)
                structure_res.append(
                    {
                        "layout_bbox": sub["box"],
                        "印章": " ".join(sorted_text),
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
            structure_res = LayoutParsingResult(
                {
                    "input_path": layout_pred["input_path"],
                    "parsing_result": structure_res,
                }
            )

            single_img_res["table_result"] = all_table_res
            single_img_res["ocr_result"] = ocr_res
            single_img_res["table_ocr_result"] = all_table_ocr_res
            single_img_res["layout_parsing_result"] = structure_res
            single_img_res["layout_parsing_result"]["page_id"] = page_id + 1
            yield VisualResult(single_img_res, page_id, inputs)

    def decode_visual_result(self, visual_result):
        ocr_text = []
        table_text_list = []
        table_html = []
        for single_img_pred in visual_result:
            layout_res = single_img_pred["layout_parsing_result"]["parsing_result"]
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

    def build_vector(
        self,
        llm_name=None,
        llm_params={},
        visual_info=None,
        min_characters=3500,
        llm_request_interval=1.0,
    ):
        """get vector for ocr"""
        if isinstance(self.llm_api, ErnieBot):
            get_vector_flag = True
        else:
            logging.warning("Do not use ErnieBot, will not get vector text.")
            get_vector_flag = False
        if not any([visual_info, self.visual_info]):
            return VectorResult({"vector": None})

        ocr_text = visual_info["ocr_text"]
        html_list = visual_info["table_html"]
        table_text_list = visual_info["table_text"]

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
        self.visual_flag = False

        return VectorResult({"vector": text_result})

    def retrieval(
        self,
        key_list,
        vector,
        llm_name=None,
        llm_params={},
        llm_request_interval=0.1,
    ):
        assert "vector" in vector
        key_list = format_key(key_list)

        # for serving
        if llm_name:
            _vector = vector["vector"]
            llm_api = create_llm_api(llm_name, llm_params)
            retrieval = llm_api.caculate_similar(
                vector=_vector,
                key_list=key_list,
                llm_params=llm_params,
                sleep_time=llm_request_interval,
            )
        else:
            _vector = vector["vector"]
            retrieval = self.llm_api.caculate_similar(
                vector=_vector, key_list=key_list, sleep_time=llm_request_interval
            )

        return RetrievalResult({"retrieval": retrieval})

    def chat(
        self,
        key_list,
        vector=None,
        visual_info=None,
        retrieval_result=None,
        user_task_description="",
        rules="",
        few_shot="",
        save_prompt=False,
        llm_name=None,
        llm_params={},
    ):
        """
        chat with key

        """
        if not any([vector, visual_info, retrieval_result]):
            return ChatResult(
                {"chat_res": "请先完成图像解析再开始再对话", "prompt": ""}
            )
        key_list = format_key(key_list)
        # first get from table, then get from text in table, last get from all ocr

        ocr_text = visual_info["ocr_text"]
        html_list = visual_info["table_html"]
        table_text_list = visual_info["table_text"]

        prompt_res = {"ocr_prompt": "str", "table_prompt": [], "html_prompt": []}

        if llm_name:
            llm_api = create_llm_api(llm_name, llm_params)
        else:
            llm_api = self.llm_api

        final_results = {}
        failed_results = ["大模型调用失败", "未知", "未找到关键信息", "None", ""]
        if html_list:
            prompt_list = self.get_prompt_for_table(
                html_list, key_list, rules, few_shot
            )
            prompt_res["html_prompt"] = prompt_list
            for prompt, table_text in zip(prompt_list, table_text_list):
                logging.debug(prompt)
                res = self.get_llm_result(llm_api, prompt)
                # TODO: why use one html but the whole table_text in next step
                if not res or list(res.values())[0] in failed_results:
                    logging.debug(
                        "table html sequence is too much longer, using ocr directly!"
                    )
                    prompt = self.get_prompt_for_ocr(
                        table_text, key_list, rules, few_shot, user_task_description
                    )
                    logging.debug(prompt)
                    prompt_res["table_prompt"].append(prompt)
                    res = self.get_llm_result(llm_api, prompt)
                for key, value in res.items():
                    if value not in failed_results and key in key_list:
                        key_list.remove(key)
                        final_results[key] = value
        if len(key_list) > 0:
            logging.debug("get result from ocr")
            if retrieval_result:
                ocr_text = retrieval_result.get("retrieval")
            elif vector:
                # for serving
                if llm_name:
                    ocr_text = self.retrieval(
                        key_list=key_list,
                        vector=vector,
                        llm_name=llm_name,
                        llm_params=llm_params,
                    )["retrieval"]
                # for local
                else:
                    ocr_text = self.retrieval(key_list=key_list, vector=vector)[
                        "retrieval"
                    ]

            prompt = self.get_prompt_for_ocr(
                ocr_text,
                key_list,
                rules,
                few_shot,
                user_task_description,
            )
            logging.debug(prompt)
            prompt_res["ocr_prompt"] = [prompt]
            res = self.get_llm_result(llm_api, prompt)
            if res:
                final_results.update(res)
        if not res and not final_results:
            final_results = {"error": llm_api.ERROR_MASSAGE}
        if save_prompt:
            return ChatResult({"chat_res": final_results, "prompt": prompt_res})
        else:
            return ChatResult({"chat_res": final_results, "prompt": ""})

    def get_llm_result(self, llm_api, prompt):
        """get llm result and decode to dict"""
        llm_result = llm_api.pred(prompt)
        # when the llm pred failed, return None
        if not llm_result:
            return {}

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
