# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""

from functools import wraps, partial

from ....utils import logging


def register(register_map, key):
    """register the option setting func
    """

    def decorator(func):
        register_map[key] = func

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class PaddleInferenceOption(object):
    """Paddle Inference Engine Option
    """
    SUPPORT_RUN_MODE = ('paddle', 'trt_fp32', 'trt_fp16', 'trt_int8', 'mkldnn',
                        'mkldnn_bf16')
    SUPPORT_DEVICE = ('gpu', 'cpu', 'npu', 'xpu', 'mlu')
    _REGISTER_MAP = {}

    register2self = partial(register, _REGISTER_MAP)

    def __init__(self, **kwargs):
        super().__init__()
        self._cfg = {}
        self._init_option(**kwargs)

    def _init_option(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._REGISTER_MAP:
                raise Exception(
                    f"{k} is not supported to set! The supported option is: \
{list(self._REGISTER_MAP.keys())}")
            self._REGISTER_MAP.get(k)(self, v)
        for k, v in self._get_default_config().items():
            self._cfg.setdefault(k, v)

    def _get_default_config(cls):
        """ get default config """
        return {
            'run_mode': 'paddle',
            'batch_size': 1,
            'device': 'gpu',
            'device_id': 0,
            'min_subgraph_size': 3,
            'shape_info_filename': None,
            'trt_calib_mode': False,
            'cpu_threads': 1,
            'trt_use_static': False
        }

    @register2self('run_mode')
    def set_run_mode(self, run_mode: str):
        """set run mode
        """
        if run_mode not in self.SUPPORT_RUN_MODE:
            support_run_mode_str = ", ".join(self.SUPPORT_RUN_MODE)
            raise ValueError(
                f"`run_mode` must be {support_run_mode_str}, but received {repr(run_mode)}."
            )
        self._cfg['run_mode'] = run_mode

    @register2self('batch_size')
    def set_batch_size(self, batch_size: int):
        """set batch size
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise Exception()
        self._cfg['batch_size'] = batch_size

    @register2self('device')
    def set_device(self, device_setting: str):
        """set device
        """
        if len(device_setting.split(":")) == 1:
            device = device_setting.split(":")[0]
            device_id = 0
        else:
            assert len(device_setting.split(":")) == 2
            device = device_setting.split(":")[0]
            device_id = device_setting.split(":")[1].split(",")[0]
            logging.warning(f"The device id has been set to {device_id}.")

        if device.lower() not in self.SUPPORT_DEVICE:
            support_run_mode_str = ", ".join(self.SUPPORT_DEVICE)
            raise ValueError(
                f"`device` must be {support_run_mode_str}, but received {repr(device)}."
            )
        self._cfg['device'] = device.lower()
        self._cfg['device_id'] = int(device_id)

    @register2self('min_subgraph_size')
    def set_min_subgraph_size(self, min_subgraph_size: int):
        """set min subgraph size
        """
        if not isinstance(min_subgraph_size, int):
            raise Exception()
        self._cfg['min_subgraph_size'] = min_subgraph_size

    @register2self('shape_info_filename')
    def set_shape_info_filename(self, shape_info_filename: str):
        """set shape info filename
        """
        self._cfg['shape_info_filename'] = shape_info_filename

    @register2self('trt_calib_mode')
    def set_trt_calib_mode(self, trt_calib_mode):
        """set trt calib mode
        """
        self._cfg['trt_calib_mode'] = trt_calib_mode

    @register2self('cpu_threads')
    def set_cpu_threads(self, cpu_threads):
        """set cpu threads
        """
        if not isinstance(cpu_threads, int) or cpu_threads < 1:
            raise Exception()
        self._cfg['cpu_threads'] = cpu_threads

    @register2self('trt_use_static')
    def set_trt_use_static(self, trt_use_static):
        """set trt use static
        """
        self._cfg['trt_use_static'] = trt_use_static

    def get_support_run_mode(self):
        """get supported run mode
        """
        return self.SUPPORT_RUN_MODE

    def get_support_device(self):
        """get supported device
        """
        return self.SUPPORT_DEVICE

    def __str__(self):
        return ",  ".join([f"{k}: {v}" for k, v in self._cfg.items()])

    def __getattr__(self, key):
        if key not in self._cfg:
            raise Exception()
        return self._cfg.get(key)
