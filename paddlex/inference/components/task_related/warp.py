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
from ..base import BaseComponent


class DocTrPostProcess(BaseComponent):
    """normalize image such as substract mean, divide std"""

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = ["doctr_img"]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {"doctr_img": "doctr_img"}

    def __init__(self, scale=None, **kwargs):
        super().__init__()
        if isinstance(scale, str):
            scale = np.float32(scale)
        self.scale = np.float32(scale if scale is not None else 255.0)

    def apply(self, pred):
        im = pred[0]
        assert isinstance(im, np.ndarray), "invalid input 'im' in DocTrPostProcess"

        im = im.squeeze()
        im = im.transpose(1, 2, 0)
        im *= self.scale
        im = im[:, :, ::-1]
        im = im.astype("uint8", copy=False)
        result = {"doctr_img": im}
        return result
