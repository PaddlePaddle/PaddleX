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

__all__ = [
    "DEBUG",
    "DRY_RUN",
    "CHECK_OPTS",
    "EAGER_INITIALIZATION",
    "INFER_BENCHMARK",
    "INFER_BENCHMARK_ITERS",
    "INFER_BENCHMARK_WARMUP",
    "INFER_BENCHMARK_OUTPUT_DIR",
    "FLAGS_json_format_model",
    "USE_PIR_TRT",
    "DISABLE_DEV_MODEL_WL",
    "DISABLE_CINN_MODEL_WL",
]


def get_flag_from_env_var(name, default, format_func=str):
    """get_flag_from_env_var"""
    env_var = os.environ.get(name, default)
    if env_var in (True, "True", "true", "TRUE", "1"):
        return True
    elif env_var in (False, "False", "false", "FALSE", "0"):
        return False
    elif env_var in (None, "None", "none", "Null", "null"):
        return None
    return format_func(env_var)


DEBUG = get_flag_from_env_var("PADDLE_PDX_DEBUG", False)
DRY_RUN = get_flag_from_env_var("PADDLE_PDX_DRY_RUN", False)
CHECK_OPTS = get_flag_from_env_var("PADDLE_PDX_CHECK_OPTS", False)
EAGER_INITIALIZATION = get_flag_from_env_var("PADDLE_PDX_EAGER_INIT", True)
FLAGS_json_format_model = get_flag_from_env_var("FLAGS_json_format_model", True)
USE_PIR_TRT = get_flag_from_env_var("PADDLE_PDX_USE_PIR_TRT", True)
DISABLE_DEV_MODEL_WL = get_flag_from_env_var("PADDLE_PDX_DISABLE_DEV_MODEL_WL", False)
DISABLE_CINN_MODEL_WL = get_flag_from_env_var("PADDLE_PDX_DISABLE_CINN_MODEL_WL", False)
DISABLE_TRT_MODEL_BL = get_flag_from_env_var("PADDLE_PDX_DISABLE_TRT_MODEL_BL", False)
DISABLE_MKLDNN_MODEL_BL = get_flag_from_env_var(
    "PADDLE_PDX_DISABLE_MKLDNN_MODEL_BL", False
)
LOCAL_FONT_FILE_PATH = get_flag_from_env_var("PADDLE_PDX_LOCAL_FONT_FILE_PATH", None)
ENABLE_MKLDNN_BYDEFAULT = get_flag_from_env_var(
    "PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", True
)

MODEL_SOURCE = os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface")


# Inference Benchmark
INFER_BENCHMARK = get_flag_from_env_var("PADDLE_PDX_INFER_BENCHMARK", False)
INFER_BENCHMARK_WARMUP = get_flag_from_env_var(
    "PADDLE_PDX_INFER_BENCHMARK_WARMUP", 0, int
)
INFER_BENCHMARK_OUTPUT_DIR = get_flag_from_env_var(
    "PADDLE_PDX_INFER_BENCHMARK_OUTPUT_DIR", None
)
INFER_BENCHMARK_ITERS = get_flag_from_env_var(
    "PADDLE_PDX_INFER_BENCHMARK_ITERS", 0, int
)
INFER_BENCHMARK_USE_CACHE_FOR_READ = get_flag_from_env_var(
    "PADDLE_PDX_INFER_BENCHMARK_USE_CACHE_FOR_READ", False
)
