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

from ...modules.image_unwarping.model_list import MODELS
from ..components import *
from ..results import DocTrResult
from ..utils.process_hook import batchable_method
from .base import BasicPredictor


class WarpPredictor(BasicPredictor):

    entities = MODELS

    def _check_args(self, kwargs):
        assert set(kwargs.keys()).issubset(set(["batch_size"]))
        return kwargs

    def _build_components(self):
        ops = {}
        ops["ReadImage"] = ReadImage(
            format="RGB", batch_size=self.kwargs.get("batch_size", 1)
        )
        ops["Normalize"] = Normalize(mean=0.0, std=1.0, scale=1.0 / 255)
        ops["ToCHWImage"] = ToCHWImage()

        predictor = ImagePredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        ops["predictor"] = predictor

        ops["postprocess"] = DocTrPostProcess()
        return ops

    @batchable_method
    def _pack_res(self, single):
        keys = ["img_path", "doctr_img"]
        return DocTrResult({key: single[key] for key in keys})
