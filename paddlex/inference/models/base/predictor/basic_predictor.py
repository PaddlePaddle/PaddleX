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

from typing import Dict, Any, Iterator
from abc import abstractmethod

from .....utils.subclass_register import AutoRegisterABCMetaClass
from .....utils.flags import (
    INFER_BENCHMARK,
    INFER_BENCHMARK_WARMUP,
    INFER_BENCHMARK_ITERS,
)
from .....utils import logging
from ....utils.pp_option import PaddlePredictorOption
from ....utils.benchmark import benchmark, ENTRY_POINT_NAME
from .base_predictor import BasePredictor


class BasicPredictor(
    BasePredictor,
    metaclass=AutoRegisterABCMetaClass,
):
    """BasicPredictor."""

    __is_base = True

    def __init__(
        self,
        model_dir: str,
        config: Dict[str, Any] = None,
        device: str = None,
        batch_size: int = 1,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        """Initializes the BasicPredictor.

        Args:
            model_dir (str): The directory where the model files are stored.
            config (Dict[str, Any], optional): The configuration dictionary. Defaults to None.
            device (str, optional): The device to run the inference engine on. Defaults to None.
            batch_size (int, optional): The batch size to predict. Defaults to 1.
            pp_option (PaddlePredictorOption, optional): The inference engine options. Defaults to None.
        """
        super().__init__(model_dir=model_dir, config=config)
        if not pp_option:
            pp_option = PaddlePredictorOption(model_name=self.model_name)
        if device:
            pp_option.set_device(device)
        trt_dynamic_shapes = (
            self.config.get("Hpi", {})
            .get("backend_configs", {})
            .get("paddle_infer", {})
            .get("trt_dynamic_shapes", None)
        )
        if trt_dynamic_shapes:
            pp_option.trt_dynamic_shapes = trt_dynamic_shapes
        trt_dynamic_shape_input_data = (
            self.config.get("Hpi", {})
            .get("backend_configs", {})
            .get("paddle_infer", {})
            .get("trt_dynamic_shape_input_data", None)
        )
        if trt_dynamic_shape_input_data:
            pp_option.trt_dynamic_shape_input_data = trt_dynamic_shape_input_data

        self.pp_option = pp_option
        self.pp_option.batch_size = batch_size
        self.batch_sampler.batch_size = batch_size

        logging.debug(f"{self.__class__.__name__}: {self.model_dir}")

    def __call__(
        self,
        input: Any,
        batch_size: int = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        **kwargs: Dict[str, Any],
    ) -> Iterator[Any]:
        """
        Predict with the input data.

        Args:
            input (Any): The input data to be predicted.
            batch_size (int, optional): The batch size to use. Defaults to None.
            device (str, optional): The device to run the predictor on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The predictor options to set. Defaults to None.
            **kwargs (Dict[str, Any]): Additional keyword arguments to set up predictor.

        Returns:
            Iterator[Any]: An iterator yielding the prediction output.
        """
        self.set_predictor(batch_size, device, pp_option)
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
        batch_size: int = None,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
    ) -> None:
        """
        Sets the predictor configuration.

        Args:
            batch_size (int, optional): The batch size to use. Defaults to None.
            device (str, optional): The device to run the predictor on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The predictor options to set. Defaults to None.

        Returns:
            None
        """
        if batch_size:
            self.batch_sampler.batch_size = batch_size
            self.pp_option.batch_size = batch_size
        if device and device != self.pp_option.device:
            self.pp_option.set_device(device)
        if pp_option and pp_option != self.pp_option:
            self.pp_option = pp_option
