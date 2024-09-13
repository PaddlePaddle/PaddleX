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

from ...utils.device import parse_device
from ....utils.func_register import FuncRegister
from ....utils import logging


class PaddlePredictorOption(object):
    """Paddle Inference Engine Option"""

    SUPPORT_RUN_MODE = (
        "paddle",
        "trt_fp32",
        "trt_fp16",
        "trt_int8",
        "mkldnn",
        "mkldnn_bf16",
    )
    SUPPORT_DEVICE = ("gpu", "cpu", "npu", "xpu", "mlu")

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, **kwargs):
        super().__init__()
        self._cfg = {}
        self._init_option(**kwargs)

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._REGISTER_MAP:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: \
{list(self._REGISTER_MAP.keys())}"
                )
            self._REGISTER_MAP.get(k)(self, v)
        for k, v in self._get_default_config().items():
            self._cfg.setdefault(k, v)

    def _get_default_config(cls):
        """get default config"""
        return {
            "run_mode": "paddle",
            "batch_size": 1,
            "device": "gpu",
            "device_id": 0,
            "min_subgraph_size": 3,
            "shape_info_filename": None,
            "trt_calib_mode": False,
            "cpu_threads": 1,
            "trt_use_static": False,
            "delete_pass": [],
        }

    @register("run_mode")
    def set_run_mode(self, run_mode: str):
        """set run mode"""
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )
        self._cfg["run_mode"] = run_mode

    @register("batch_size")
    def set_batch_size(self, batch_size: int):
        """set batch size"""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception()
        self._cfg["batch_size"] = batch_size

    @register("device")
    def set_device(self, device: str):
        """set device"""
        device_type, device_ids = parse_device(device)
        self._cfg["device"] = device_type
        if device_type not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"The device type must be one of {support_run_mode_str}, but received {repr(device_type)}."
            )
        device_id = device_ids[0] if device_ids is not None else 0
        self._cfg["device_id"] = device_id
        logging.warning(f"The device ID has been set to {device_id}.")

    @register("min_subgraph_size")
    def set_min_subgraph_size(self, min_subgraph_size: int):
        """set min subgraph size"""
        if not isinstance(min_subgraph_size, int):
            raise Exception()
        self._cfg["min_subgraph_size"] = min_subgraph_size

    @register("shape_info_filename")
    def set_shape_info_filename(self, shape_info_filename: str):
        """set shape info filename"""
        self._cfg["shape_info_filename"] = shape_info_filename

    @register("trt_calib_mode")
    def set_trt_calib_mode(self, trt_calib_mode):
        """set trt calib mode"""
        self._cfg["trt_calib_mode"] = trt_calib_mode

    @register("cpu_threads")
    def set_cpu_threads(self, cpu_threads):
        """set cpu threads"""
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._cfg["cpu_threads"] = cpu_threads

    @register("trt_use_static")
    def set_trt_use_static(self, trt_use_static):
        """set trt use static"""
        self._cfg["trt_use_static"] = trt_use_static

    @register("delete_pass")
    def set_delete_pass(self, delete_pass):
        self._cfg["delete_pass"] = delete_pass

    def get_support_run_mode(self):
        """get supported run mode"""
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device"""
        return self.SUPPORT_DEVICE

    def get_device(self):
        """get device"""
        return f"{self._cfg['device']}:{self._cfg['device_id']}"

    def __str__(self):
        return ",  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception(f"The key ({key}) is not found in cfg: \n {self._cfg}")
        return self._cfg.get(key)
