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

import tempfile
from typing import Any, Dict, List

import ultra_infer as ui
import numpy as np
from paddlex.inference.common.batch_sampler import ImageBatchSampler
from paddlex.inference.models.text_recognition.result import TextRecResult
from paddlex.modules.text_recognition.model_list import MODELS

from paddlex_hpi.models.base import CVPredictor


class TextRecPredictor(CVPredictor):
    entities = MODELS

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TextRecResult

    def _build_ui_model(self, option: ui.RuntimeOption) -> ui.vision.ocr.Recognizer:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt") as f:
            pp_config = self.config["PostProcess"]
            for lab in pp_config["character_dict"]:
                f.write(lab + "\n")
            f.flush()
            model = ui.vision.ocr.Recognizer(
                str(self.model_path),
                str(self.params_path),
                label_path=f.name,
                runtime_option=option,
            )
            self._config_ui_preprocessor(model)
        return model

    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        batch_raw_imgs = self._data_reader(imgs=batch_data.instances)
        imgs = [np.ascontiguousarray(img) for img in batch_raw_imgs]
        ui_results = self._ui_model.batch_predict(imgs)

        texts_list = ui_results.text
        rec_score_list = ui_results.rec_scores

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "rec_text": texts_list,
            "rec_score": rec_score_list,
        }

    def _config_ui_preprocessor(self, model: ui.vision.ocr.Recognizer) -> None:
        pp_config = self.config["PreProcess"]
        preprocessor = model.preprocessor
        found_resize_op = False
        for item in pp_config["transform_ops"]:
            op_name = next(iter(item))
            op_config = item[op_name]
            if op_name == "DecodeImage":
                if op_config["channel_first"]:
                    raise RuntimeError(
                        "`DecodeImage.channel_first` must be set to False."
                    )
            elif op_name == "RecResizeImg":
                preprocessor.rec_image_shape = op_config["image_shape"]
                found_resize_op = True
            elif op_name == "MultiLabelEncode":
                pass
            elif op_name == "KeepKeys":
                pass
            else:
                raise RuntimeError(f"Unkown preprocessing operator: {op_name}")
        if not found_resize_op:
            raise RuntimeError("Could not find the config for `RecResizeImg`.")
