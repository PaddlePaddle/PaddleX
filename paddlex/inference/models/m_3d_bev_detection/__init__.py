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

from ....modules.m_3d_bev_detection.model_list import MODELS
from ....utils.import_guard import import_paddle
from ..bindings import create_binding_registration, register_predictor_binding_map
from .predictor import BEVDet3DRunnerPredictor


def _build_bevfusion_static_runner(
    *,
    model_name,
    model_dir,
    model_config,
    engine_config,
    default_builder=None,
):
    if not callable(default_builder):
        raise RuntimeError("Default paddle_static runner builder is unavailable.")

    paddle = import_paddle()
    if not paddle.is_compiled_with_cuda() or paddle.is_compiled_with_rocm():
        raise RuntimeError("3D BEVFusion custom ops only support CUDA GPU platform.")

    from ....ops.iou3d_nms import nms_gpu  # noqa: F401
    from ....ops.voxelize import hard_voxelize  # noqa: F401

    return default_builder(
        model_name=model_name,
        model_dir=model_dir,
        model_config=model_config,
        engine_config=engine_config,
    )


register_predictor_binding_map(
    BEVDet3DRunnerPredictor,
    {
        "paddle_static": create_binding_registration(
            MODELS,
            runner_builder=_build_bevfusion_static_runner,
        ),
        "hpi": MODELS,
    },
)

# Backward compatibility
BEVDet3DPredictor = BEVDet3DRunnerPredictor
