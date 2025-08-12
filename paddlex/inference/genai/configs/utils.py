# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import yaml


def load_backend_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def update_backend_config(config, overrides):
    for k, v in overrides.items():
        config[k] = v


def set_config_defaults(config, defaults):
    for k, v in defaults.items():
        if k not in config:
            config[k] = v


def backend_config_to_args(config):
    # Limited support
    args = []
    for k, v in config.items():
        opt = "--" + k
        args.append(opt)
        if not isinstance(v, bool):
            args.append(str(v))
    return args
