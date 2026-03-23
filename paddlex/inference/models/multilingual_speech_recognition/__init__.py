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

import numpy as np

from .... import constants
from ..bindings import create_binding_registration, register_predictor_binding_map
from ..engines.paddle import PaddleDynamicEngine
from ..runners import PaddleDynamicRunner
from ..runners.paddle_dynamic_runner import resolve_paddle_runner_device
from .predictor import WhisperRunnerPredictor


def _build_whisper_runner(model_name, model_dir, model_config, engine_config):
    del model_name, model_config
    import paddle

    from ....utils.device import TemporaryDeviceChanger
    from .processors import ModelDimensions, Whisper

    with TemporaryDeviceChanger(resolve_paddle_runner_device(engine_config)):
        model_file = (model_dir / f"{constants.MODEL_FILE_PREFIX}.pdparams").as_posix()
        model_dict = paddle.load(model_file)
        dims = ModelDimensions(**model_dict["dims"])
        model = Whisper(dims)
        model.load_dict(model_dict)
        model.eval()

    resource_path = model_dir.as_posix()

    def infer_fn(model, inputs):
        result = model.transcribe(
            inputs["mel"],
            resource_path=resource_path,
            **inputs["decode_kwargs"],
        )
        return np.array([result], dtype=object)

    return PaddleDynamicRunner(
        model,
        config=engine_config,
        infer_fn=infer_fn,
    )


register_predictor_binding_map(
    WhisperRunnerPredictor,
    {
        "paddle_dynamic": create_binding_registration(
            WhisperRunnerPredictor.entities,
            **{
                PaddleDynamicEngine.BINDING_EXTRA_RUNNER_BUILDER_KEY: _build_whisper_runner,
            },
        ),
    },
)

# Backward compatibility
WhisperPredictor = WhisperRunnerPredictor
