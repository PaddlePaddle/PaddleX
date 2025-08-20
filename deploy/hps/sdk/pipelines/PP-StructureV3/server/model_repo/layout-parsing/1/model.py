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
        visualize_enabled = input.visualize if input.visualize is not None else self.app_config.visualize

        file_bytes = utils.get_raw_bytes(input.file)
        images, data_info = utils.file_to_images(
            file_bytes,
            file_type,
            max_num_imgs=self.context["max_num_input_imgs"],
        )

        result = list(
            self.pipeline(
                images,
                use_doc_orientation_classify=input.useDocOrientationClassify,
                use_doc_unwarping=input.useDocUnwarping,
                use_textline_orientation=input.useTextlineOrientation,
                use_seal_recognition=input.useSealRecognition,
                use_table_recognition=input.useTableRecognition,
                use_formula_recognition=input.useFormulaRecognition,
                use_chart_recognition=input.useChartRecognition,
                use_region_detection=input.useRegionDetection,
                layout_threshold=input.layoutThreshold,
                layout_nms=input.layoutNms,
                layout_unclip_ratio=input.layoutUnclipRatio,
                layout_merge_bboxes_mode=input.layoutMergeBboxesMode,
                text_det_limit_side_len=input.textDetLimitSideLen,
                text_det_limit_type=input.textDetLimitType,
                text_det_thresh=input.textDetThresh,
                text_det_box_thresh=input.textDetBoxThresh,
                text_det_unclip_ratio=input.textDetUnclipRatio,
                text_rec_score_thresh=input.textRecScoreThresh,
                seal_det_limit_side_len=input.sealDetLimitSideLen,
                seal_det_limit_type=input.sealDetLimitType,
                seal_det_thresh=input.sealDetThresh,
                seal_det_box_thresh=input.sealDetBoxThresh,
                seal_det_unclip_ratio=input.sealDetUnclipRatio,
                seal_rec_score_thresh=input.sealRecScoreThresh,
                use_wired_table_cells_trans_to_html=input.useWiredTableCellsTransToHtml,
                use_wireless_table_cells_trans_to_html=input.useWirelessTableCellsTransToHtml,
                use_table_orientation_classify=input.useTableOrientationClassify,
                use_ocr_results_with_table_cells=input.useOcrResultsWithTableCells,
                use_e2e_wired_table_rec_model=input.useE2eWiredTableRecModel,
                use_e2e_wireless_table_rec_model=input.useE2eWirelessTableRecModel,
            )
        )

        layout_parsing_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
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
