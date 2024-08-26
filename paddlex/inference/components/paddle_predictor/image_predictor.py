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

import numpy as np

from .base_predictor import BasePaddlePredictor


class ImagePredictor(BasePaddlePredictor):

    def to_batch(self, imgs):
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]

    def format_output(self, output):
        return [{"pred": np.array(res)} for res in output[0].tolist()]
