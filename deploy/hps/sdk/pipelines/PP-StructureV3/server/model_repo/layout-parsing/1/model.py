# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
from typing import Any, Dict, Final, List, Tuple

from paddlex_hps_server import (
    BaseTritonPythonModel,
    app_common,
    protocol,
    schemas,
    utils,
)
from paddlex_hps_server.storage import SupportsGetURL, create_storage

_DEFAULT_MAX_NUM_INPUT_IMGS: Final[int] = 10
_DEFAULT_MAX_OUTPUT_IMG_SIZE: Final[Tuple[int, int]] = (2000, 2000)


class _SequentialExecutor(object):
    def map(self, fn, *iterables):
        return map(fn, *iterables)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class TritonPythonModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        self.context["file_storage"] = None
        self.context["return_img_urls"] = False
        self.context["max_num_input_imgs"] = _DEFAULT_MAX_NUM_INPUT_IMGS
        self.context["max_output_img_size"] = _DEFAULT_MAX_OUTPUT_IMG_SIZE
        if self.app_config.extra:
            if "file_storage" in self.app_config.extra:
                self.context["file_storage"] = create_storage(
                    self.app_config.extra["file_storage"]
                )
            if "return_img_urls" in self.app_config.extra:
                self.context["return_img_urls"] = self.app_config.extra[
                    "return_img_urls"
                ]
            if "max_num_input_imgs" in self.app_config.extra:
                self.context["max_num_input_imgs"] = self.app_config.extra[
                    "max_num_input_imgs"
                ]
            if "max_output_img_size" in self.app_config.extra:
                self.context["max_output_img_size"] = self.app_config.extra[
                    "max_output_img_size"
                ]
        if self.context["return_img_urls"]:
            file_storage = self.context["file_storage"]
            if not file_storage:
                raise ValueError(
                    "The file storage must be properly configured when URLs need to be returned."
                )
            if not isinstance(file_storage, SupportsGetURL):
                raise TypeError(f"{type(file_storage)} does not support getting URLs.")

    def get_input_model_type(self):
        return schemas.pp_structurev3.InferRequest

    def get_result_model_type(self):
        return schemas.pp_structurev3.InferResult

    def run(self, input, log_id):
        return self.run_batch([input], [log_id], log_id)

    def run_batch(self, inputs, log_ids, batch_id):
        result_or_output_dic = {}

        input_groups = self._group_inputs(inputs)

        max_group_size = max(map(len, input_groups))
        if max_group_size > 1:
            executor = ThreadPoolExecutor(max_workers=max_group_size)
        else:
            executor = _SequentialExecutor()

        with executor:
            for input_group in input_groups:
                input_ids_g = list(map(itemgetter(0), input_group))
                inputs_g = list(map(itemgetter(1), input_group))

                log_ids_g = [log_ids[i] for i in input_ids_g]

                ret = executor.map(self._preprocess, inputs_g, log_ids_g)
                ind_img_lsts, ind_data_info_lst, ind_visualize_enabled_lst = [], [], []
                for i, item in enumerate(ret):
                    if isinstance(item, tuple):
                        assert len(item) == 3, len(item)
                        ind_img_lsts.append(item[0])
                        ind_data_info_lst.append(item[1])
                        ind_visualize_enabled_lst.append(item[2])
                    else:
                        input_id = input_ids_g[i]
                        result_or_output_dic[input_id] = item

                if len(ind_img_lsts):
                    images = [img for item in ind_img_lsts for img in item]
                    preds = list(
                        self.pipeline(
                            images,
                            use_doc_orientation_classify=inputs_g[
                                0
                            ].useDocOrientationClassify,
                            use_doc_unwarping=inputs_g[0].useDocUnwarping,
                            use_textline_orientation=inputs_g[0].useTextlineOrientation,
                            use_seal_recognition=inputs_g[0].useSealRecognition,
                            use_table_recognition=inputs_g[0].useTableRecognition,
                            use_formula_recognition=inputs_g[0].useFormulaRecognition,
                            use_chart_recognition=inputs_g[0].useChartRecognition,
                            use_region_detection=inputs_g[0].useRegionDetection,
                            format_block_content=inputs_g[0].formatBlockContent,
                            layout_threshold=inputs_g[0].layoutThreshold,
                            layout_nms=inputs_g[0].layoutNms,
                            layout_unclip_ratio=inputs_g[0].layoutUnclipRatio,
                            layout_merge_bboxes_mode=inputs_g[0].layoutMergeBboxesMode,
                            text_det_limit_side_len=inputs_g[0].textDetLimitSideLen,
                            text_det_limit_type=inputs_g[0].textDetLimitType,
                            text_det_thresh=inputs_g[0].textDetThresh,
                            text_det_box_thresh=inputs_g[0].textDetBoxThresh,
                            text_det_unclip_ratio=inputs_g[0].textDetUnclipRatio,
                            text_rec_score_thresh=inputs_g[0].textRecScoreThresh,
                            seal_det_limit_side_len=inputs_g[0].sealDetLimitSideLen,
                            seal_det_limit_type=inputs_g[0].sealDetLimitType,
                            seal_det_thresh=inputs_g[0].sealDetThresh,
                            seal_det_box_thresh=inputs_g[0].sealDetBoxThresh,
                            seal_det_unclip_ratio=inputs_g[0].sealDetUnclipRatio,
                            seal_rec_score_thresh=inputs_g[0].sealRecScoreThresh,
                            use_wired_table_cells_trans_to_html=inputs_g[
                                0
                            ].useWiredTableCellsTransToHtml,
                            use_wireless_table_cells_trans_to_html=inputs_g[
                                0
                            ].useWirelessTableCellsTransToHtml,
                            use_table_orientation_classify=inputs_g[
                                0
                            ].useTableOrientationClassify,
                            use_ocr_results_with_table_cells=inputs_g[
                                0
                            ].useOcrResultsWithTableCells,
                            use_e2e_wired_table_rec_model=inputs_g[
                                0
                            ].useE2eWiredTableRecModel,
                            use_e2e_wireless_table_rec_model=inputs_g[
                                0
                            ].useE2eWirelessTableRecModel,
                        )
                    )

                    if len(preds) != len(images):
                        raise RuntimeError(
                            f"The number of predictions ({len(preds)}) is not the same as the number of input images ({len(images)})."
                        )

                    start_idx = 0
                    ind_preds = []
                    for item in ind_img_lsts:
                        ind_preds.append(preds[start_idx : start_idx + len(item)])
                        start_idx += len(item)

                    for i, result in zip(
                        input_ids_g,
                        executor.map(
                            self._postprocess,
                            ind_img_lsts,
                            ind_data_info_lst,
                            ind_visualize_enabled_lst,
                            ind_preds,
                            log_ids_g,
                            inputs_g,
                        ),
                    ):
                        result_or_output_dic[i] = result

            assert len(result_or_output_dic) == len(
                inputs
            ), f"Expected {len(inputs)} results or outputs, but got {len(result_or_output_dic)}"

            return [result_or_output_dic[i] for i in range(len(inputs))]

    def _group_inputs(self, inputs):
        def _to_hashable(obj):
            if isinstance(obj, list):
                return tuple(obj)
            elif isinstance(obj, dict):
                return tuple(sorted(obj.items()))
            else:
                return obj

        def _hash(input):
            return hash(
                tuple(
                    map(
                        _to_hashable,
                        (
                            input.useDocOrientationClassify,
                            input.useDocUnwarping,
                            input.useTextlineOrientation,
                            input.useSealRecognition,
                            input.useTableRecognition,
                            input.useFormulaRecognition,
                            input.useChartRecognition,
                            input.useRegionDetection,
                            input.formatBlockContent,
                            input.layoutThreshold,
                            input.layoutNms,
                            input.layoutUnclipRatio,
                            input.layoutMergeBboxesMode,
                            input.textDetLimitSideLen,
                            input.textDetLimitType,
                            input.textDetThresh,
                            input.textDetBoxThresh,
                            input.textDetUnclipRatio,
                            input.textRecScoreThresh,
                            input.sealDetLimitSideLen,
                            input.sealDetLimitType,
                            input.sealDetThresh,
                            input.sealDetBoxThresh,
                            input.sealDetUnclipRatio,
                            input.sealRecScoreThresh,
                            input.useWiredTableCellsTransToHtml,
                            input.useWirelessTableCellsTransToHtml,
                            input.useTableOrientationClassify,
                            input.useOcrResultsWithTableCells,
                            input.useE2eWiredTableRecModel,
                            input.useE2eWirelessTableRecModel,
                        ),
                    )
                )
            )

        groups = {}
        for i, inp in enumerate(inputs):
            group_key = _hash(inp)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((i, inp))

        return list(groups.values())

    def _preprocess(self, input, log_id):
        if input.fileType is None:
            if utils.is_url(input.file):
                maybe_file_type = utils.infer_file_type(input.file)
                if maybe_file_type is None or not (
                    maybe_file_type == "PDF" or maybe_file_type == "IMAGE"
                ):
                    return protocol.create_aistudio_output_without_result(
                        422,
                        "Unsupported file type",
                        log_id=log_id,
                    )
                file_type = maybe_file_type
            else:
                return protocol.create_aistudio_output_without_result(
                    422,
                    "File type cannot be determined",
                    log_id=log_id,
                )
        else:
            file_type = "PDF" if input.fileType == 0 else "IMAGE"
        visualize_enabled = (
            input.visualize
            if input.visualize is not None
            else self.app_config.visualize
        )

        file_bytes = utils.get_raw_bytes(input.file)
        images, data_info = utils.file_to_images(
            file_bytes,
            file_type,
            max_num_imgs=self.context["max_num_input_imgs"],
        )

        return images, data_info, visualize_enabled

    def _postprocess(self, images, data_info, visualize_enabled, preds, log_id, input):
        layout_parsing_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, preds)):
            pruned_res = app_common.prune_result(item.json["res"])
            md_data = item.markdown
            md_text = md_data["markdown_texts"]
            md_imgs = app_common.postprocess_images(
                md_data["markdown_images"],
                log_id,
                filename_template=f"markdown_{i}/{{key}}",
                file_storage=self.context["file_storage"],
                return_urls=self.context["return_img_urls"],
                max_img_size=self.context["max_output_img_size"],
            )
            md_flags = md_data["page_continuation_flags"]
            if visualize_enabled:
                imgs = {
                    "input_img": img,
                    **item.img,
                }
                imgs = app_common.postprocess_images(
                    imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=self.context["file_storage"],
                    return_urls=self.context["return_img_urls"],
                    max_img_size=self.context["max_output_img_size"],
                )
            else:
                imgs = {}
            layout_parsing_results.append(
                dict(
                    prunedResult=pruned_res,
                    markdown=dict(
                        text=md_text,
                        images=md_imgs,
                        isStart=md_flags[0],
                        isEnd=md_flags[1],
                    ),
                    outputImages=(
                        {k: v for k, v in imgs.items() if k != "input_img"}
                        if imgs
                        else None
                    ),
                    inputImage=imgs.get("input_img"),
                )
            )

        return schemas.pp_structurev3.InferResult(
            layoutParsingResults=layout_parsing_results,
            dataInfo=data_info,
        )
