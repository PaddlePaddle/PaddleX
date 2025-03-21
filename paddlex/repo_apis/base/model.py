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


import abc
import contextlib
import functools
import inspect
import os
import tempfile

from ...utils import flags, logging
from ...utils.cache import get_cache_dir
from ...utils.device import parse_device
from ...utils.errors import UnsupportedAPIError, UnsupportedParamError
from ...utils.misc import CachedProperty as cached_property
from .config import Config
from .register import (
    build_model_from_model_info,
    build_runner_from_model_info,
    get_registered_model_info,
)

__all__ = ["PaddleModel", "BaseModel"]


def _create_model(model_name=None, config=None):
    """_create_model"""
    if model_name is None and config is None:
        raise ValueError("At least one of `model_name` and `config` must be not None.")
    elif model_name is not None and config is not None:
        if model_name != config.model_name:
            raise ValueError(
                "If both `model_name` and `config` are not None, `model_name` should be the same as \
`config.model_name`."
            )
    elif model_name is None and config is not None:
        model_name = config.model_name
    try:
        model_info = get_registered_model_info(model_name)
    except KeyError as e:
        raise UnsupportedParamError(
            f"{repr(model_name)} is not a registered model name."
        ) from e
    return build_model_from_model_info(model_info=model_info, config=config)


PaddleModel = _create_model


class BaseModel(metaclass=abc.ABCMeta):
    """
    Abstract base class of Model.

    Model defines how Config and Runner interact with each other. In addition,
    Model provides users with multiple APIs to perform model training,
    prediction, etc.
    """

    _API_FULL_LIST = ("train", "evaluate", "predict", "export", "infer", "compression")
    _API_SUPPORTED_OPTS_KEY_PATTERN = "supported_{api_name}_opts"

    def __init__(self, model_name, config=None):
        """
        Initialize the instance.

        Args:
            model_name (str): A registered model name.
            config (base.config.BaseConfig|None): Config. Default: None.
        """
        super().__init__()

        self.name = model_name
        self.model_info = get_registered_model_info(model_name)
        # NOTE: We build runner instance here by extracting runner info from model info
        # so that we don't have to overwrite the `__init__` method of each child class.
        self.runner = build_runner_from_model_info(self.model_info)
        if config is None:
            logging.warning(
                "We strongly discourage leaving `config` unset or setting it to None. "
                "Please note that when `config` is None, default settings will be used for every unspecified \
configuration item, "
                "which may lead to unexpected result. Please make sure that this is what you intend to do."
            )
            config = Config(model_name)
        self.config = config

        self._patch_apis()

    @abc.abstractmethod
    def train(
        self,
        batch_size=None,
        learning_rate=None,
        epochs_iters=None,
        ips=None,
        device="gpu",
        resume_path=None,
        dy2st=False,
        amp="OFF",
        num_workers=None,
        use_vdl=True,
        save_dir=None,
        **kwargs,
    ):
        """
        Train a model.

        Args:
            batch_size (int|None): Number of samples in each mini-batch. If
                multiple devices are used, this is the batch size on each device.
                If None, use a default setting. Default: None.
            learning_rate (float|None): Learning rate of model training. If
                None, use a default setting. Default: None.
            epochs_iters (int|None): Total epochs or iterations of model
                training. If None, use a default setting. Default: None.
            ips (str|None): If not None, enable multi-machine training mode.
                `ips` specifies Paddle cluster node ips, e.g.,
                '192.168.0.16,192.168.0.17'. Default: None.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'gpu', 'gpu:1,2'. Default: 'gpu'.
            resume_path (str|None): If not None, resume training from the model
                snapshot corresponding to the weight file `resume_path`. If
                None, use a default setting. Default: None.
            dy2st (bool): Whether to enable dynamic-to-static training.
                Default: False.
            amp (str): Optimization level to use in AMP training. Choices are
                ['O1', 'O2', 'OFF']. Default: 'OFF'.
            num_workers (int|None): Number of subprocesses to use for data
                loading. If None, use a default setting. Default: None.
            use_vdl (bool): Whether to enable VisualDL during training.
                Default: True.
            save_dir (str|None): Directory to store model snapshots and logs. If
                None, use a default setting. Default: None.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        weight_path,
        batch_size=None,
        ips=None,
        device="gpu",
        amp="OFF",
        num_workers=None,
        **kwargs,
    ):
        """
        Evaluate a model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            batch_size (int|None): Number of samples in each mini-batch. If
                multiple devices are used, this is the batch size on each device.
                If None, use a default setting. Default: None.
            ips (str|None): If not None, enable multi-machine evaluation mode.
                `ips` specifies Paddle cluster node ips, e.g.,
                '192.168.0.16,192.168.0.17'. Default: None.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'gpu', 'gpu:1,2'. Default: 'gpu'.
            amp (str): Optimization level to use in AMP training. Choices are
                ['O1', 'O2', 'OFF']. Default: 'OFF'.
            num_workers (int|None): Number of subprocesses to use for data
                loading. If None, use a default setting. Default: None.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, weight_path, input_path, device="gpu", save_dir=None, **kwargs):
        """
        Make prediction with a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            input_path (str): Path of the input file, e.g. an image.
            device (str): A string that describes the device to use, e.g.,
                'cpu', 'gpu'. Default: 'gpu'.
            save_dir (str|None): Directory to store prediction results. If None,
                use a default setting. Default: None.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, weight_path, save_dir, **kwargs):
        """
        Export a pre-trained model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            save_dir (str): Directory to store the exported model.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, model_dir, input_path, device="gpu", save_dir=None, **kwargs):
        """
        Make inference with an exported inference model.

        Args:
            model_dir (str): Path of the exported inference model.
            input_path (str): Path of the input file, e.g. an image.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'gpu'. Default: 'gpu'.
            save_dir (str|None): Directory to store inference results. If None,
                use a default setting. Default: None.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(
        self,
        weight_path,
        batch_size=None,
        learning_rate=None,
        epochs_iters=None,
        device="gpu",
        use_vdl=True,
        save_dir=None,
        **kwargs,
    ):
        """
        Perform quantization aware training (QAT) and export the quantized
        model.

        Args:
            weight_path (str): Path of the weights to initialize the model.
            batch_size (int|None): Number of samples in each mini-batch. If
                multiple devices are used, this is the batch size on each
                device. If None, use a default setting. Default: None.
            learning_rate (float|None): Learning rate of QAT. If None, use a
                default setting. Default: None.
            epochs_iters (int|None): Total epochs of iterations of model
                training. If None, use a default setting. Default: None.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'gpu'. Default: 'gpu'.
            use_vdl (bool): Whether to enable VisualDL during training.
                Default: True.
            save_dir (str|None): Directory to store the results. If None, use a
                default setting. Default: None.

        Returns:
            tuple[paddlex.repo_apis.base.utils.subprocess.CompletedProcess]
        """
        raise NotImplementedError

    @contextlib.contextmanager
    def _create_new_config_file(self):
        cls = self.__class__
        model_name = self.model_info["model_name"]
        tag = "_".join([cls.__name__.lower(), model_name])
        yaml_file_name = tag + ".yml"
        if not flags.DEBUG:
            with tempfile.TemporaryDirectory(dir=get_cache_dir()) as td:
                path = os.path.join(td, yaml_file_name)
                with open(path, "w", encoding="utf-8"):
                    pass
                yield path
        else:
            path = os.path.join(get_cache_dir(), yaml_file_name)
            with open(path, "w", encoding="utf-8"):
                pass
            yield path

    @contextlib.contextmanager
    def _create_new_val_json_file(self):
        cls = self.__class__
        model_name = self.model_info["model_name"]
        tag = "_".join([cls.__name__.lower(), model_name])
        json_file_name = tag + "_test.json"
        if not flags.DEBUG:
            with tempfile.TemporaryDirectory(dir=get_cache_dir()) as td:
                path = os.path.join(td, json_file_name)
                with open(path, "w", encoding="utf-8"):
                    pass
                yield path
        else:
            path = os.path.join(get_cache_dir(), json_file_name)
            with open(path, "w", encoding="utf-8"):
                pass
            yield path

    @cached_property
    def supported_apis(self):
        """supported apis"""
        return self.model_info.get("supported_apis", None)

    @cached_property
    def supported_train_opts(self):
        """supported train opts"""
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name="train"), None
        )

    @cached_property
    def supported_evaluate_opts(self):
        """supported evaluate opts"""
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name="evaluate"), None
        )

    @cached_property
    def supported_predict_opts(self):
        """supported predcit opts"""
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name="predict"), None
        )

    @cached_property
    def supported_infer_opts(self):
        """supported infer opts"""
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name="infer"), None
        )

    @cached_property
    def supported_compression_opts(self):
        """supported copression opts"""
        return self.model_info.get(
            self._API_SUPPORTED_OPTS_KEY_PATTERN.format(api_name="compression"), None
        )

    @cached_property
    def supported_dataset_types(self):
        """supported dataset types"""
        return self.model_info.get("supported_dataset_types", None)

    @staticmethod
    def _assert_empty_kwargs(kwargs):
        if len(kwargs) > 0:
            # For compatibility
            logging.warning(f"Unconsumed keyword arguments detected: {kwargs}.")

            # raise RuntimeError(
            #     f"Unconsumed keyword arguments detected: {kwargs}.")

    def _patch_apis(self):
        def _make_unavailable(bnd_method):
            @functools.wraps(bnd_method)
            def _unavailable_api(*args, **kwargs):
                model_name = self.name
                api_name = bnd_method.__name__
                raise UnsupportedAPIError(
                    f"{model_name} does not support `{api_name}`."
                )

            return _unavailable_api

        def _add_prechecks(bnd_method):
            @functools.wraps(bnd_method)
            def _api_with_prechecks(*args, **kwargs):
                sig = inspect.Signature.from_callable(bnd_method)
                bnd_args = sig.bind(*args, **kwargs)
                args_dict = bnd_args.arguments
                # Merge default values
                for p in sig.parameters.values():
                    if p.name not in args_dict and p.default is not p.empty:
                        args_dict[p.name] = p.default

                # Rely on nonlocal variable `checks`
                for check in checks:
                    # We throw any unhandled exception
                    check.check(args_dict)

                return bnd_method(*args, **kwargs)

            api_name = bnd_method.__name__
            checks = []
            # We hardcode the prechecks for each API here
            if api_name == "train":
                opts = self.supported_train_opts
                if opts is not None:
                    if "device" in opts:
                        checks.append(_CheckDevice(opts["device"], check_mc=True))
                    if "dy2st" in opts:
                        checks.append(_CheckDy2St(opts["dy2st"]))
                    if "amp" in opts:
                        checks.append(_CheckAMP(opts["amp"]))
            elif api_name == "evaluate":
                opts = self.supported_evaluate_opts
                if opts is not None:
                    if "device" in opts:
                        checks.append(_CheckDevice(opts["device"], check_mc=True))
                    if "amp" in opts:
                        checks.append(_CheckAMP(opts["amp"]))
            elif api_name == "predict":
                opts = self.supported_predict_opts
                if opts is not None:
                    if "device" in opts:
                        checks.append(_CheckDevice(opts["device"], check_mc=False))
            elif api_name == "infer":
                opts = self.supported_infer_opts
                if opts is not None:
                    if "device" in opts:
                        checks.append(_CheckDevice(opts["device"], check_mc=False))
            elif api_name == "compression":
                opts = self.supported_compression_opts
                if opts is not None:
                    if "device" in opts:
                        checks.append(_CheckDevice(opts["device"], check_mc=True))
            else:
                return bnd_method

            return _api_with_prechecks

        supported_apis = self.supported_apis
        if supported_apis is not None:
            avail_api_set = set(self.supported_apis)
        else:
            avail_api_set = set(self._API_FULL_LIST)
        for api_name in self._API_FULL_LIST:
            api = getattr(self, api_name)
            if api_name not in avail_api_set:
                # We decorate real API implementation with `_make_unavailable`
                # so that an error is always raised when the API is called.
                decorated_api = _make_unavailable(api)
                # Monkey-patch
                setattr(self, api_name, decorated_api)
            else:
                if flags.CHECK_OPTS:
                    # We decorate real API implementation with `_add_prechecks`
                    # to perform sanity and validity checks before invoking the
                    # internal API.
                    decorated_api = _add_prechecks(api)
                    setattr(self, api_name, decorated_api)


class _CheckFailed(Exception):
    """_CheckFailed"""

    # Allow `_CheckFailed` class to be recognized using `hasattr(exc, 'check_failed_error')`
    check_failed_error = True

    def __init__(self, arg_name, arg_val, legal_vals):
        self.arg_name = arg_name
        self.arg_val = arg_val
        self.legal_vals = legal_vals

    def __str__(self):
        return f"`{self.arg_name}` is expected to be one of or conforms to {self.legal_vals}, but got {self.arg_val}"


class _APICallArgsChecker(object):
    """_APICallArgsChecker"""

    def __init__(self, legal_vals):
        super().__init__()
        self.legal_vals = legal_vals

    def check(self, args):
        """check"""
        raise NotImplementedError


class _CheckDevice(_APICallArgsChecker):
    """_CheckDevice"""

    def __init__(self, legal_vals, check_mc=False):
        super().__init__(legal_vals)
        self.check_mc = check_mc

    def check(self, args):
        """check"""
        assert "device" in args
        device = args["device"]
        if device is not None:
            device_type, dev_ids = parse_device(device)
            if not self.check_mc:
                if device_type not in self.legal_vals:
                    raise _CheckFailed("device", device, self.legal_vals)
            else:
                # Currently we only check multi-device settings for GPUs
                if device_type != "gpu":
                    if device_type not in self.legal_vals:
                        raise _CheckFailed("device", device, self.legal_vals)
                else:
                    n1c1_desc = f"{device_type}_n1c1"
                    n1cx_desc = f"{device_type}_n1cx"
                    nxcx_desc = f"{device_type}_nxcx"

                    if len(dev_ids) <= 1:
                        if (
                            n1c1_desc not in self.legal_vals
                            and n1cx_desc not in self.legal_vals
                            and nxcx_desc not in self.legal_vals
                        ):
                            raise _CheckFailed("device", device, self.legal_vals)
                    else:
                        assert "ips" in args
                        if args["ips"] is not None:
                            # Multi-machine
                            if nxcx_desc not in self.legal_vals:
                                raise _CheckFailed("device", device, self.legal_vals)
                        else:
                            # Single-machine multi-device
                            if (
                                n1cx_desc not in self.legal_vals
                                and nxcx_desc not in self.legal_vals
                            ):
                                raise _CheckFailed("device", device, self.legal_vals)
        else:
            # When `device` is None, we assume that a default device that the
            # current model supports will be used, so we simply do nothing.
            pass


class _CheckDy2St(_APICallArgsChecker):
    """_CheckDy2St"""

    def check(self, args):
        """check"""
        assert "dy2st" in args
        dy2st = args["dy2st"]
        if isinstance(self.legal_vals, list):
            assert len(self.legal_vals) == 1
            support_dy2st = bool(self.legal_vals[0])
        else:
            support_dy2st = bool(self.legal_vals)
        if dy2st is not None:
            if dy2st and not support_dy2st:
                raise _CheckFailed("dy2st", dy2st, [support_dy2st])
        else:
            pass


class _CheckAMP(_APICallArgsChecker):
    """_CheckAMP"""

    def check(self, args):
        """check"""
        assert "amp" in args
        amp = args["amp"]
        if amp is not None:
            if amp != "OFF" and amp not in self.legal_vals:
                raise _CheckFailed("amp", amp, self.legal_vals)
        else:
            pass
