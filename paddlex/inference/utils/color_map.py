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


def get_colormap(rgb=False):
    """
    Get colormap
    """
    color_list = np.array(
        [
            0xFF,
            0x00,
            0x00,
            0xCC,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0x66,
            0x00,
            0x66,
            0xFF,
            0xCC,
            0x00,
            0xFF,
            0xFF,
            0x4D,
            0x00,
            0x80,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xB2,
            0x00,
            0x1A,
            0xFF,
            0xFF,
            0x00,
            0xE5,
            0xFF,
            0x99,
            0x00,
            0x33,
            0xFF,
            0x00,
            0x00,
            0xFF,
            0xFF,
            0x33,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x99,
            0xFF,
            0xE5,
            0x00,
            0x00,
            0xFF,
            0x1A,
            0x00,
            0xB2,
            0xFF,
            0x80,
            0x00,
            0xFF,
            0xFF,
            0x00,
            0x4D,
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3))
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list.astype("int32")
