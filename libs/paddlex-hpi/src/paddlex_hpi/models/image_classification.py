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
from paddlex.inference.models_new.image_classification.result import TopkResult
from paddlex.modules.image_classification.model_list import MODELS
from pydantic import BaseModel

from paddlex_hpi.models.base import CVPredictor, HPIParams


class _ClasPPParams(BaseModel):
    topk: int
    label_list: Optional[List[str]] = None


class ClasPredictor(CVPredictor):
    entities = MODELS

    def __init__(
        self,
        model_dir: Union[str, os.PathLike],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        use_onnx_model: Optional[bool] = None,
        hpi_params: Optional[HPIParams] = None,
        topk: Union[int, None] = None,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            config=config,
            device=device,
            use_onnx_model=use_onnx_model,
            hpi_params=hpi_params,
        )
        self._pp_params = self._get_pp_params()
        self._topk = topk or self._pp_params.topk

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TopkResult

    def _build_ui_model(
        self, option: ui.RuntimeOption
    ) -> ui.vision.classification.PaddleClasModel:
        if self._onnx_format:
            model = ui.vision.classification.PaddleClasModel(
                str(self.model_path),
                str(self.params_path),
                str(self.config_path),
                runtime_option=option,
                model_format=ui.ModelFormat.ONNX,
            )
        else:
            model = ui.vision.classification.PaddleClasModel(
                str(self.model_path),
                str(self.params_path),
                str(self.config_path),
                runtime_option=option,
            )
        return model

    def process(
        self, batch_data: List[Any], topk: Union[int, None] = None
    ) -> Dict[str, List[Any]]:
        batch_raw_imgs = self._data_reader(imgs=batch_data)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        self._ui_model.postprocessor.topk = topk or self._topk
        ui_results = self._ui_model.batch_predict(imgs)

        class_ids_list = []
        scores_list = []
        label_names_list = []
        for ui_result in ui_results:
            class_ids_list.append(ui_result.label_ids)
            scores_list.append(np.around(ui_result.scores, decimals=5).tolist())
            if self._pp_params.label_list is not None:
                label_names_list.append(
                    [self._pp_params.label_list[i] for i in ui_result.label_ids]
                )

        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "class_ids": class_ids_list,
            "scores": scores_list,
            "label_names": label_names_list,
        }

    def _get_pp_params(self) -> _ClasPPParams:
        pp_config = self.config["PostProcess"]
        if "Topk" not in pp_config:
            raise RuntimeError("`Topk` config not found")
        topk_config = pp_config["Topk"]
        topk = topk_config["topk"]
        label_list = topk_config.get("label_list", None)
        return _ClasPPParams(topk=topk, label_list=label_list)
