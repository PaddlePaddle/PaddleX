# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
import paddle.fluid as fluid
import os
import sys
import paddlex as pdx
import paddlex.utils.logging as logging

__all__ = ['export_onnx']


def export_onnx(model_dir, save_dir, fixed_input_shape):
    assert len(fixed_input_shape) == 2, "len of fixed input shape must == 2"
    model = pdx.load_model(model_dir, fixed_input_shape)
    model_name = os.path.basename(model_dir.strip('/')).split('/')[-1]
    export_onnx_model(model, save_dir)


def export_onnx_model(model, save_dir):
    if model.model_type == "detector" or model.__class__.__name__ == "FastSCNN":
        logging.error(
            "Only image classifier models and semantic segmentation models(except FastSCNN) are supported to export to ONNX"
        )
    try:
        import x2paddle
        if x2paddle.__version__ < '0.7.4':
            logging.error("You need to upgrade x2paddle >= 0.7.4")
    except:
        logging.error(
            "You need to install x2paddle first, pip install x2paddle>=0.7.4")
    from x2paddle.op_mapper.paddle_op_mapper import PaddleOpMapper
    mapper = PaddleOpMapper()
    mapper.convert(model.test_prog, save_dir)
