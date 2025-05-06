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

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import ValidationError

from ..... import constants
from .....utils import logging
from .....utils.deps import require_hpip
from .....utils.device import get_default_device, parse_device
from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_ITERS,
    INFER_BENCHMARK_WARMUP,
)
from .....utils.subclass_register import AutoRegisterABCMetaClass
from ....common.batch_sampler import BaseBatchSampler
from ....utils.benchmark import ENTRY_POINT_NAME, benchmark
from ....utils.hpi import HPIConfig, HPIInfo
from ....utils.io import YAMLReader
from ....utils.pp_option import PaddlePredictorOption
from ...common import HPInfer, PaddleInfer


class PredictionWrap:
    """Wraps the prediction data and supports get by index."""

    def __init__(self, data: Dict[str, List[Any]], num: int) -> None:
        """Initializes the PredictionWrap with prediction data.

        Args:
            data (Dict[str, List[Any]]): A dictionary where keys are string identifiers and values are lists of predictions.
            num (int): The number of predictions, that is length of values per key in the data dictionary.

        Raises:
            AssertionError: If the length of any list in data does not match num.
        """
        assert isinstance(data, dict), "data must be a dictionary"
        for k in data:
            assert len(data[k]) == num, f"{len(data[k])} != {num} for key {k}!"
        self._data = data
        self._keys = data.keys()

    def get_by_idx(self, idx: int) -> Dict[str, Any]:
        """Get the prediction by specified index.

        Args:
            idx (int): The index to get predictions from.

        Returns:
            Dict[str, Any]: A dictionary with the same keys as the input data, but with the values at the specified index.
        """
        return {key: self._data[key][idx] for key in self._keys}


class BasePredictor(
    ABC,
    metaclass=AutoRegisterABCMetaClass,
):
    MODEL_FILE_PREFIX = constants.MODEL_FILE_PREFIX

    __is_base = True

    def __init__(
        self,
        model_dir: str,
        config: Optional[Dict[str, Any]] = None,
        *,
        device: Optional[str] = None,
        batch_size: int = 1,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the BasePredictor.

        Args:
            model_dir (str): The directory where the model files are stored.
            config (Optional[Dict[str, Any]], optional): The model configuration
                dictionary. Defaults to None.
            device (Optional[str], optional): The device to run the inference
                engine on. Defaults to None.
            batch_size (int, optional): The batch size to predict.
                Defaults to 1.
            pp_option (Optional[PaddlePredictorOption], optional): The inference
                engine options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference
                plugin. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The high-performance inference configuration dictionary.
                Defaults to None.
        """
        super().__init__()

        self.model_dir = Path(model_dir)
        self.config = config if config else self.load_config(self.model_dir)
        self.batch_sampler = self._build_batch_sampler()
        self.result_class = self._get_result_class()

        # alias predict() to the __call__()
        self.predict = self.__call__

        self.batch_sampler.batch_size = batch_size
        self._use_hpip = use_hpip
        if not use_hpip:
            self._pp_option = self._prepare_pp_option(pp_option, device)
        else:
            require_hpip()
            self._hpi_config = self._prepare_hpi_config(hpi_config, device)

        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

    @property
    def config_path(self) -> str:
        """
        Get the path to the configuration file.

        Returns:
            str: The path to the configuration file.
        """
        return self.get_config_path(self.model_dir)

    @property
    def model_name(self) -> str:
        """
        Get the model name.

        Returns:
            str: The model name.
        """
        return self.config["Global"]["model_name"]

    @property
    def pp_option(self) -> PaddlePredictorOption:
        if not hasattr(self, "_pp_option"):
            raise AttributeError(f"{repr(self)} has no attribute 'pp_option'.")
        return self._pp_option

    @property
    def hpi_config(self) -> HPIConfig:
        if not hasattr(self, "_hpi_config"):
            raise AttributeError(f"{repr(self)} has no attribute 'hpi_config'.")
        return self._hpi_config

    @property
    def use_hpip(self) -> bool:
        return self._use_hpip

    def __call__(
        self,
        input: Any,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """
        Predict with the input data.

        Args:
            input (Any): The input data to be predicted.
            batch_size (int, optional): The batch size to use. Defaults to None.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(batch_size)
        if INFER_BENCHMARK:
            # TODO(zhang-prog): Get metadata of input data
            @benchmark.timeit_with_options(name=ENTRY_POINT_NAME)
            def _apply(input, **kwargs):
                return list(self.apply(input, **kwargs))

            if isinstance(input, list):
                raise TypeError("`input` cannot be a list in benchmark mode")
            input = [input] * batch_size

            if not (INFER_BENCHMARK_WARMUP > 0 or INFER_BENCHMARK_ITERS > 0):
                raise RuntimeError(
                    "At least one of `INFER_BENCHMARK_WARMUP` and `INFER_BENCHMARK_ITERS` must be greater than zero"
                )

            if INFER_BENCHMARK_WARMUP > 0:
                benchmark.start_warmup()
                for _ in range(INFER_BENCHMARK_WARMUP):
                    output = _apply(input, **kwargs)
                benchmark.collect(batch_size)
                benchmark.stop_warmup()

            if INFER_BENCHMARK_ITERS > 0:
                for _ in range(INFER_BENCHMARK_ITERS):
                    output = _apply(input, **kwargs)
                benchmark.collect(batch_size)

            yield output[0]
        else:
            yield from self.apply(input, **kwargs)

    def set_predictor(
        self,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (Optional[int], optional): The batch size to use. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size

    def get_hpi_info(self):
        if "Hpi" not in self.config:
            return None
        try:
            return HPIInfo.model_validate(self.config["Hpi"])
        except ValidationError as e:
            logging.exception("The HPI info in the model config file is invalid.")
            raise RuntimeError(f"Invalid HPI info: {str(e)}") from e

    def create_static_infer(self):
        if not self._use_hpip:
            return PaddleInfer(self.model_dir, self.MODEL_FILE_PREFIX, self._pp_option)
        else:
            return HPInfer(
                self.model_dir,
                self.MODEL_FILE_PREFIX,
                self._hpi_config,
            )

    def apply(self, input: Any, **kwargs) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """
        if INFER_BENCHMARK:
            if not isinstance(input, list):
                raise TypeError("In benchmark mode, `input` must be a list")
            batches = list(self.batch_sampler(input))
            if len(batches) != 1 or len(batches[0]) != len(input):
                raise ValueError("Unexpected number of instances")
        else:
            batches = self.batch_sampler(input)
        for batch_data in batches:
            prediction = self.process(batch_data, **kwargs)
            prediction = PredictionWrap(prediction, len(batch_data))
            for idx in range(len(batch_data)):
                yield self.result_class(prediction.get_by_idx(idx))

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        """process the batch data sampled from BatchSampler and return the prediction result.

        Args:
            batch_data (List[Any]): The batch data sampled from BatchSampler.

        Returns:
            Dict[str, List[Any]]: The prediction result.
        """
        raise NotImplementedError

    @classmethod
    def get_config_path(cls, model_dir) -> str:
        """Get the path to the configuration file for the given model directory.

        Args:
            model_dir (Path): The directory where the static model files is stored.

        Returns:
            Path: The path to the configuration file.
        """
        return model_dir / f"{cls.MODEL_FILE_PREFIX}.yml"

    @classmethod
    def load_config(cls, model_dir) -> Dict:
        """Load the configuration from the specified model directory.

        Args:
            model_dir (Path): The where the static model files is stored.

        Returns:
            dict: The loaded configuration dictionary.
        """
        yaml_reader = YAMLReader()
        return yaml_reader.read(cls.get_config_path(model_dir))

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        """Build batch sampler.

        Returns:
            BaseBatchSampler: batch sampler object.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        """Get the result class.

        Returns:
            type: The result class.
        """
        raise NotImplementedError

    def _prepare_pp_option(
        self,
        pp_option: Optional[PaddlePredictorOption],
        device: Optional[str],
    ) -> PaddlePredictorOption:
        if pp_option is None or device is not None:
            device_info = self._get_device_info(device)
        else:
            device_info = None
        if pp_option is None:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        elif pp_option.model_name is None:
            pp_option.model_name = self.model_name
        if device_info:
            pp_option.device_type = device_info[0]
            pp_option.device_id = device_info[1]
        hpi_info = self.get_hpi_info()
        if hpi_info is not None:
            hpi_info = hpi_info.model_dump(exclude_unset=True)
            if pp_option.trt_dynamic_shapes is None:
                trt_dynamic_shapes = (
                    hpi_info.get("backend_configs", {})
                    .get("paddle_infer", {})
                    .get("trt_dynamic_shapes", None)
                )
                if trt_dynamic_shapes is not None:
                    logging.debug(
                        "TensorRT dynamic shapes set to %s", trt_dynamic_shapes
                    )
                    pp_option.trt_dynamic_shapes = trt_dynamic_shapes
            if pp_option.trt_dynamic_shape_input_data is None:
                trt_dynamic_shape_input_data = (
                    hpi_info.get("backend_configs", {})
                    .get("paddle_infer", {})
                    .get("trt_dynamic_shape_input_data", None)
                )
                if trt_dynamic_shape_input_data is not None:
                    logging.debug(
                        "TensorRT dynamic shape input data set to %s",
                        trt_dynamic_shape_input_data,
                    )
                    pp_option.trt_dynamic_shape_input_data = (
                        trt_dynamic_shape_input_data
                    )
        return pp_option

    def _prepare_hpi_config(
        self,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]],
        device: Optional[str],
    ) -> HPIConfig:
        if hpi_config is None:
            hpi_config = {}
        elif isinstance(hpi_config, HPIConfig):
            hpi_config = hpi_config.model_dump(exclude_unset=True)
        else:
            hpi_config = deepcopy(hpi_config)

        if "model_name" not in hpi_config:
            hpi_config["model_name"] = self.model_name

        if device is not None or "device_type" not in hpi_config:
            device_type, device_id = self._get_device_info(device)
            hpi_config["device_type"] = device_type
            if device is not None or "device_id" not in hpi_config:
                hpi_config["device_id"] = device_id

        if "hpi_info" not in hpi_config:
            hpi_info = self.get_hpi_info()
            if hpi_info is not None:
                hpi_config["hpi_info"] = hpi_info

        hpi_config = HPIConfig.model_validate(hpi_config)

        return hpi_config

    # Should this be static?
    def _get_device_info(self, device):
        if device is None:
            device = get_default_device()
        device_type, device_ids = parse_device(device)
        if device_ids is not None:
            device_id = device_ids[0]
        else:
            device_id = None
        if device_ids and len(device_ids) > 1:
            logging.debug("Got multiple device IDs. Using the first one: %d", device_id)
        return device_type, device_id
