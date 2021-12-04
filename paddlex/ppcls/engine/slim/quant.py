# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import paddle
from paddlex.ppcls.utils import logger

QUANT_CONFIG = {
    # weight preprocess type, default is None and no preprocessing is performed.
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}


def get_quaner(config, model):
    if config.get("Slim", False) and config["Slim"].get("quant", False):
        from paddleslim.dygraph.quant import QAT
        assert config["Slim"]["quant"]["name"].lower(
        ) == 'pact', 'Only PACT quantization method is supported now'
        QUANT_CONFIG["activation_preprocess_type"] = "PACT"
        quanter = QAT(config=QUANT_CONFIG)
        quanter.quantize(model)
        logger.info("QAT model summary:")
        paddle.summary(model, (1, 3, 224, 224))
    else:
        quanter = None
    return quanter
