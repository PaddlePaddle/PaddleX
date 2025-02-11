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
from typing import Any, Dict, List, Optional, Union

import ultra_infer as ui
import numpy as np
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.models.text_detection.result import TextDetResult
from paddlex.modules.text_detection.model_list import CURVE_MODELS, MODELS
from paddlex.utils import logging

from paddlex_hpi._utils.misc import parse_scale
from paddlex_hpi.models.base import CVPredictor, HPIParams


class TextDetPredictor(CVPredictor):
    entities = MODELS

    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        hpi_params: Optional[HPIParams] = None,
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        max_candidates: Union[int, None] = None,
        unclip_ratio: Union[float, None] = None,
        use_dilation: Union[bool, None] = None,
    ) -> None:
        if limit_type is not None:
            logging.warning(
                "The default value for `limit_type` is max, and cannot be set in PaddleX HPI."
            )
        if max_candidates is not None:
            logging.warning(
                "The default value for `max_candidates` is 1000, and cannot be set in PaddleX HPI."
            )
        super().__init__(
            model_dir=model_dir,
            config=config,
            device=device,
            batch_size=batch_size,
            hpi_params=hpi_params,
        )
        self._limit_side_len = limit_side_len or self._max_side_len
        self._thresh = thresh or self._changeable_params["thresh"]
        self._box_thresh = box_thresh or self._changeable_params["thresh"]
        self._unclip_ratio = unclip_ratio or self._changeable_params["unclip_ratio"]
        self._use_dilation = use_dilation or self._changeable_params["use_dilation"]

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TextDetResult

    # HACK
    @property
    def _is_curve_model(self) -> bool:
        return self.model_name in CURVE_MODELS

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> Union[ui.vision.ocr.DBDetector, ui.vision.ocr.DBCURVEDetector]:
        if self._is_curve_model:
            model = ui.vision.ocr.DBCURVEDetector(
                str(self.model_path),
                str(self.params_path),
                runtime_option=option,
            )
        else:
            model = ui.vision.ocr.DBDetector(
                str(self.model_path),
                str(self.params_path),
                runtime_option=option,
            )
        self._config_ui_preprocessor()
        self._config_ui_postprocessor()
        return model

    def process(
        self,
        batch_data: List[Any],
        limit_side_len: Union[int, None] = None,
        limit_type: Union[str, None] = None,
        thresh: Union[float, None] = None,
        box_thresh: Union[float, None] = None,
        max_candidates: Union[int, None] = None,
        unclip_ratio: Union[float, None] = None,
        use_dilation: Union[bool, None] = None,
    ) -> Dict[str, List[Any]]:
        if limit_type is not None:
            logging.warning(
                "The default value for `limit_type` is max, and cannot be set in PaddleX HPI."
            )
        if max_candidates is not None:
            logging.warning(
                "The default value for `max_candidates` is 1000, and cannot be set in PaddleX HPI."
            )
        self._ui_model.preprocessor.set_normalize(self._mean, self._std, True)
        self._ui_model.preprocessor.max_side_len = (
            limit_side_len or self._limit_side_len
        )
        postprocessor = self._ui_model.postprocessor
        postprocessor.det_db_thresh = thresh or self._thresh
        postprocessor.det_db_box_thresh = box_thresh or self._box_thresh
        postprocessor.det_db_unclip_ratio = unclip_ratio or self._unclip_ratio
        postprocessor.use_dilation = use_dilation or self._use_dilation
        postprocessor.det_db_score_mode = self._changeable_params["score_mode"]
        if self._is_curve_model:
            if self._changeable_params["box_type"] not in ("quad", "poly"):
                raise RuntimeError("Invalid value of `DBPostProcess.box_type`.")
            if self._changeable_params["box_type"] == "quad":
                postprocessor.det_db_box_type = "bbox"
            else:
                postprocessor.det_db_box_type = "poly"

        batch_raw_imgs = self._data_reader(imgs=batch_data.instances)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        ui_results = self._ui_model.batch_predict(imgs)

        dt_polys_list = []
        dt_scores_list = []
        for ui_result in ui_results:
            polys = [np.array(list(zip(*([iter(box)] * 2)))) for box in ui_result.boxes]
            dt_polys_list.append(polys)
            # XXX: Currently, we cannot get scores from `ui_result`, so we
            # temporarily use dummy scores here.
            dummy_scores = [0.0 for _ in ui_result.boxes]
            dt_scores_list.append(dummy_scores)
        # breakpoint()
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "dt_polys": dt_polys_list,
            "dt_scores": dt_scores_list,
        }

    def _config_ui_preprocessor(self) -> None:
        pp_config = self.config["PreProcess"]
        for item in pp_config["transform_ops"]:
            op_name = next(iter(item))
            op_config = item[op_name]
            # XXX: Default values copied from
            # `paddlex.inference.models.TextDetPredictor`
            if op_name == "DecodeImage":
                if op_config["channel_first"]:
                    raise RuntimeError(
                        "`DecodeImage.channel_first` must be set to False."
                    )
            elif op_name == "DetResizeForTest":
                self._max_side_len = op_config.get("resize_long", 960)
            elif op_name == "NormalizeImage":
                if "scale" in op_config and not (
                    abs(parse_scale(op_config["scale"]) - 1 / 255) < 1e-9
                ):
                    raise RuntimeError("`NormalizeImage.scale` must be set to 1/255.")
                if "channel_num" in op_config and op_config["channel_num"] != 3:
                    raise RuntimeError("`NormalizeImage.channel_num` must be set to 3.")

                self._mean = op_config.get("mean", [0.485, 0.456, 0.406])
                self._std = op_config.get("std", [0.229, 0.224, 0.225])

            elif op_name == "ToCHWImage":
                # Do nothing
                pass
            elif op_name == "DetLabelEncode":
                pass
            elif op_name == "KeepKeys":
                pass
            else:
                raise RuntimeError(f"Unkown preprocessing operator: {op_name}")

    def _config_ui_postprocessor(self) -> None:
        pp_config = self.config["PostProcess"]
        # XXX: Default values copied from
        # `paddlex.inference.models.TextDetPredictor`
        self._changeable_params: Dict[str, Any] = {
            "thresh": 0.3,
            "box_thresh": 0.7,
            "unclip_ratio": 2.0,
            "score_mode": "fast",
            "use_dilation": False,
        }
        self._unchangeable_params: Dict[str, Any] = {
            "max_candidates": 1000,
            "box_type": "quad",
        }
        if self._is_curve_model:
            self._changeable_params["box_type"] = self._unchangeable_params.pop(
                "box_type"
            )
        if "name" in pp_config and pp_config["name"] == "DBPostProcess":
            for name in self._changeable_params:
                if name in pp_config:
                    self._changeable_params[name] = pp_config[name]
            for name, val in self._unchangeable_params.items():
                if name in pp_config and pp_config[name] != val:
                    raise RuntimeError(
                        f"`DBPostProcess.{name}` must be set to {repr(val)}."
                    )
        else:
            raise RuntimeError("Invalid config")
