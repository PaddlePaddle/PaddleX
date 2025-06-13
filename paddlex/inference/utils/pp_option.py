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
from copy import deepcopy
from typing import Dict, List

from ...utils import logging
from ...utils.device import (
    check_supported_device_type,
    get_default_device,
    parse_device,
    set_env_for_device_type,
)
from ...utils.flags import (
    DISABLE_MKLDNN_MODEL_BL,
    DISABLE_TRT_MODEL_BL,
    ENABLE_MKLDNN_BYDEFAULT,
    USE_PIR_TRT,
)
from .misc import is_mkldnn_available
from .mkldnn_blocklist import MKLDNN_BLOCKLIST
from .new_ir_blocklist import NEWIR_BLOCKLIST
from .trt_blocklist import TRT_BLOCKLIST
from .trt_config import TRT_CFG_SETTING, TRT_PRECISION_MAP


def get_default_run_mode(model_name, device_type):
    if not model_name:
        return "paddle"
    if device_type != "cpu":
        return "paddle"
    if (
        ENABLE_MKLDNN_BYDEFAULT
        and is_mkldnn_available()
        and model_name not in MKLDNN_BLOCKLIST
    ):
        return "mkldnn"
    else:
        return "paddle"


class PaddlePredictorOption(object):
    """Paddle Inference Engine Option"""

    # NOTE: TRT modes start with `trt_`
    SUPPORT_RUN_MODE = (
        "paddle",
        "paddle_fp32",
        "paddle_fp16",
        "trt_fp32",
        "trt_fp16",
        "trt_int8",
        "mkldnn",
        "mkldnn_bf16",
    )
    SUPPORT_DEVICE = ("gpu", "cpu", "npu", "xpu", "mlu", "dcu", "gcu")

    def __init__(self, model_name=None, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._cfg = {}
        self._init_option(**kwargs)
        self._changed = False

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name

    @property
    def changed(self):
        return self._changed

    @changed.setter
    def changed(self, v):
        assert isinstance(v, bool)
        self._changed = v

    def copy(self):
        obj = type(self)(self._model_name)
        obj._cfg = deepcopy(self._cfg)
        if hasattr(self, "trt_cfg_setting"):
            obj.trt_cfg_setting = self.trt_cfg_setting
        return obj

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if self._has_setter(k):
                setattr(self, k, v)
            else:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: {self._get_settable_attributes()}"
                )
        for k, v in self._get_default_config().items():
            self._cfg.setdefault(k, v)

        # for trt
        if self.run_mode in ("trt_int8", "trt_fp32", "trt_fp16"):
            trt_cfg_setting = TRT_CFG_SETTING[self.model_name]
            if USE_PIR_TRT:
                trt_cfg_setting["precision_mode"] = TRT_PRECISION_MAP[self.run_mode]
            else:
                trt_cfg_setting["enable_tensorrt_engine"]["precision_mode"] = (
                    TRT_PRECISION_MAP[self.run_mode]
                )
            self.trt_cfg_setting = trt_cfg_setting

    def _get_default_config(self):
        """get default config"""
        device_type, device_ids = parse_device(get_default_device())

        default_config = {
            "run_mode": get_default_run_mode(self.model_name, device_type),
            "device_type": device_type,
            "device_id": None if device_ids is None else device_ids[0],
            "cpu_threads": 8,
            "delete_pass": [],
            "enable_new_ir": True if self.model_name not in NEWIR_BLOCKLIST else False,
            "enable_cinn": False,
            "trt_cfg_setting": {},
            "trt_use_dynamic_shapes": True,  # only for trt
            "trt_collect_shape_range_info": True,  # only for trt
            "trt_discard_cached_shape_range_info": False,  # only for trt
            "trt_dynamic_shapes": None,  # only for trt
            "trt_dynamic_shape_input_data": None,  # only for trt
            "trt_shape_range_info_path": None,  # only for trt
            "trt_allow_rebuild_at_runtime": True,  # only for trt
            "mkldnn_cache_capacity": 10,
        }
        return default_config

    def _update(self, k, v):
        self._cfg[k] = v
        self.changed = True

    @property
    def run_mode(self):
        return self._cfg["run_mode"]

    @run_mode.setter
    def run_mode(self, run_mode: str):
        """set run mode"""
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )

        if run_mode.startswith("mkldnn") and not is_mkldnn_available():
            logging.warning("MKL-DNN is not available. Using `paddle` instead.")
            run_mode = "paddle"

        # TODO: Check if trt is available

        if self._model_name is not None:
            # TRT Blocklist
            if (
                not DISABLE_TRT_MODEL_BL
                and run_mode.startswith("trt")
                and self._model_name in TRT_BLOCKLIST
            ):
                logging.warning(
                    f"The model({self._model_name}) is not supported to run in trt mode! Using `paddle` instead!"
                )
                run_mode = "paddle"
            # MKLDNN Blocklist
            elif (
                not DISABLE_MKLDNN_MODEL_BL
                and run_mode.startswith("mkldnn")
                and self._model_name in MKLDNN_BLOCKLIST
            ):
                logging.warning(
                    f"The model({self._model_name}) is not supported to run in MKLDNN mode! Using `paddle` instead!"
                )
                run_mode = "paddle"

        self._update("run_mode", run_mode)

    @property
    def device_type(self):
        return self._cfg["device_type"]

    @device_type.setter
    def device_type(self, device_type):
        if device_type not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"The device type must be one of {support_run_mode_str}, but received {repr(device_type)}."
            )
        check_supported_device_type(device_type, self.model_name)
        self._update("device_type", device_type)
        set_env_for_device_type(device_type)
        # XXX(gaotingquan): set flag to accelerate inference in paddle 3.0b2
        if device_type in ("gpu", "cpu"):
            os.environ["FLAGS_enable_pir_api"] = "1"

    @property
    def device_id(self):
        return self._cfg["device_id"]

    @device_id.setter
    def device_id(self, device_id):
        self._update("device_id", device_id)

    @property
    def cpu_threads(self):
        return self._cfg["cpu_threads"]

    @cpu_threads.setter
    def cpu_threads(self, cpu_threads):
        """set cpu threads"""
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._update("cpu_threads", cpu_threads)

    @property
    def delete_pass(self):
        return self._cfg["delete_pass"]

    @delete_pass.setter
    def delete_pass(self, delete_pass):
        self._update("delete_pass", delete_pass)

    @property
    def enable_new_ir(self):
        return self._cfg["enable_new_ir"]

    @enable_new_ir.setter
    def enable_new_ir(self, enable_new_ir: bool):
        """set run mode"""
        self._update("enable_new_ir", enable_new_ir)

    @property
    def enable_cinn(self):
        return self._cfg["enable_cinn"]

    @enable_cinn.setter
    def enable_cinn(self, enable_cinn: bool):
        """set run mode"""
        self._update("enable_cinn", enable_cinn)

    @property
    def trt_cfg_setting(self):
        return self._cfg["trt_cfg_setting"]

    @trt_cfg_setting.setter
    def trt_cfg_setting(self, config: Dict):
        """set trt config"""
        assert isinstance(
            config, dict
        ), f"The trt_cfg_setting must be `dict` type, but received `{type(config)}` type!"
        self._update("trt_cfg_setting", config)

    @property
    def trt_use_dynamic_shapes(self):
        return self._cfg["trt_use_dynamic_shapes"]

    @trt_use_dynamic_shapes.setter
    def trt_use_dynamic_shapes(self, trt_use_dynamic_shapes):
        self._update("trt_use_dynamic_shapes", trt_use_dynamic_shapes)

    @property
    def trt_collect_shape_range_info(self):
        return self._cfg["trt_collect_shape_range_info"]

    @trt_collect_shape_range_info.setter
    def trt_collect_shape_range_info(self, trt_collect_shape_range_info):
        self._update("trt_collect_shape_range_info", trt_collect_shape_range_info)

    @property
    def trt_discard_cached_shape_range_info(self):
        return self._cfg["trt_discard_cached_shape_range_info"]

    @trt_discard_cached_shape_range_info.setter
    def trt_discard_cached_shape_range_info(self, trt_discard_cached_shape_range_info):
        self._update(
            "trt_discard_cached_shape_range_info", trt_discard_cached_shape_range_info
        )

    @property
    def trt_dynamic_shapes(self):
        return self._cfg["trt_dynamic_shapes"]

    @trt_dynamic_shapes.setter
    def trt_dynamic_shapes(self, trt_dynamic_shapes: Dict[str, List[List[int]]]):
        assert isinstance(trt_dynamic_shapes, dict)
        for input_k in trt_dynamic_shapes:
            assert isinstance(trt_dynamic_shapes[input_k], list)
        self._update("trt_dynamic_shapes", trt_dynamic_shapes)

    @property
    def trt_dynamic_shape_input_data(self):
        return self._cfg["trt_dynamic_shape_input_data"]

    @trt_dynamic_shape_input_data.setter
    def trt_dynamic_shape_input_data(
        self, trt_dynamic_shape_input_data: Dict[str, List[float]]
    ):
        self._update("trt_dynamic_shape_input_data", trt_dynamic_shape_input_data)

    @property
    def trt_shape_range_info_path(self):
        return self._cfg["trt_shape_range_info_path"]

    @trt_shape_range_info_path.setter
    def trt_shape_range_info_path(self, trt_shape_range_info_path: str):
        """set shape info filename"""
        self._update("trt_shape_range_info_path", trt_shape_range_info_path)

    @property
    def trt_allow_rebuild_at_runtime(self):
        return self._cfg["trt_allow_rebuild_at_runtime"]

    @trt_allow_rebuild_at_runtime.setter
    def trt_allow_rebuild_at_runtime(self, trt_allow_rebuild_at_runtime):
        self._update("trt_allow_rebuild_at_runtime", trt_allow_rebuild_at_runtime)

    @property
    def mkldnn_cache_capacity(self):
        return self._cfg["mkldnn_cache_capacity"]

    @mkldnn_cache_capacity.setter
    def mkldnn_cache_capacity(self, capacity: int):
        self._update("mkldnn_cache_capacity", capacity)

    # For backward compatibility
    # TODO: Issue deprecation warnings
    @property
    def shape_info_filename(self):
        return self.trt_shape_range_info_path

    @shape_info_filename.setter
    def shape_info_filename(self, shape_info_filename):
        self.trt_shape_range_info_path = shape_info_filename

    def set_device(self, device: str):
        """set device"""
        if not device:
            return
        device_type, device_ids = parse_device(device)
        self.device_type = device_type
        device_id = device_ids[0] if device_ids is not None else None
        self.device_id = device_id
        if device_ids is None or len(device_ids) > 1:
            logging.debug(f"The device ID has been set to {device_id}.")

    def get_support_run_mode(self):
        """get supported run mode"""
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device"""
        return self.SUPPORT_DEVICE

    def __str__(self):
        return ",  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception(f"The key ({key}) is not found in cfg: \n {self._cfg}")
        return self._cfg.get(key)

    def __eq__(self, obj):
        if isinstance(obj, PaddlePredictorOption):
            return obj._cfg == self._cfg
        return False

    def _has_setter(self, attr):
        prop = getattr(self.__class__, attr, None)
        return isinstance(prop, property) and prop.fset is not None

    def _get_settable_attributes(self):
        return [
            name
            for name, prop in vars(self.__class__).items()
            if isinstance(prop, property) and prop.fset is not None
        ]
