# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ....utils import logging

CINN_WHITELIST = [
    "CLIP_vit_base_patch16_224",
    "MobileNetV2_x1_0",
    "PP-HGNet_small",
    "PP-LCNet_x1_0",
    "ResNet50",
    "Deeplabv3-R50",
    "Deeplabv3_Plus-R50",
    "PP-LiteSeg-T",
    "SLANet",
    "PP-OCRv4_mobile_rec",
    "PP-OCRv4_server_rec",
    "Nonstationary_ad",
    "PatchTST_ad",
    "TimesNet_ad",
    "Nonstationary",
    "PatchTST",
    "RLinear",
    "TiDE",
    "TimesNet",
    "CLIP_vit_large_patch14_224",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "PP-HGNet_tiny",
    "PP-HGNet_base",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "ResNet18",
    "ResNet34",
    "ResNet101",
    "ResNet152",
    "ResNet18_vd",
    "ResNet34_vd",
    "ResNet50_vd",
    "ResNet101_vd",
    "ResNet152_vd",
    "ResNet200_vd",
    "Deeplabv3-R101",
    "Deeplabv3_Plus-R101",
    "PP-LiteSeg-B",
    "SLANet_plus",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_server_rec_doc",
]


# TODO(gaotingquan): paddle v3.0.0 don't support enable CINN easily
def enable_cinn_backend():
    import paddle

    if not paddle.is_compiled_with_cinn():
        logging.debug(
            "Your paddle is not compiled with CINN, can not use CINN backend."
        )
        return

    # equivalent to `FLAGS_prim_all=1`
    paddle.base.core._set_prim_all_enabled(True)
    # equivalent to `FLAGS_prim_enable_dynamic=1`
    paddle.base.framework.set_flags({"FLAGS_prim_enable_dynamic": True})
    os.environ["FLAGS_prim_enable_dynamic"] = "1"
    # equivalent to `FLAGS_use_cinn=1`
    paddle.base.framework.set_flags({"FLAGS_use_cinn": True})
    os.environ["FLAGS_use_cinn"] = "1"
