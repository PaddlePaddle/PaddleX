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


import os
from ...repo_apis.base import Config, PaddleModel
from ...utils.device import get_device


def build_model(model_name: str, device: str = None, config_path: str = None) -> tuple:
    """build Config and PaddleModel

    Args:
        model_name (str): model name
        device (str): device, such as gpu, cpu, npu, xpu, mlu
        config_path (str, optional): path to the PaddleX config yaml file.
            Defaults to None, i.e. using the default config file.

    Returns:
        tuple(Config, PaddleModel): the Config and PaddleModel
    """
    config = Config(model_name, config_path)

    if device:
        config.update_device(get_device(device))
    model = PaddleModel(config=config)
    return config, model
