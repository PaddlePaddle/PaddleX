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

import abc
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    Optional,
    TypedDict,
    Union,
)

import ultra_infer as ui
from ultra_infer.model import BaseUltraInferModel
from paddlex.inference.common.reader import ReadImage, ReadTS
from paddlex.inference.models import BasePredictor
from paddlex.inference.utils.new_ir_blacklist import NEWIR_BLOCKLIST
from paddlex.utils import device as device_helper
from paddlex.utils import logging
from paddlex.utils.subclass_register import AutoRegisterABCMetaClass
from typing_extensions import assert_never

from paddlex_hpi._config import HPIConfig
from paddlex_hpi._utils.typing import Backend

HPI_CONFIG_KEY: Final[str] = "Hpi"


class HPIParams(TypedDict, total=False):
    config: Dict[str, Any]


class HPPredictor(BasePredictor, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    def __init__(
        self,
        model_dir: Union[str, PathLike],
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        use_onnx_model: Optional[bool] = None,
        hpi_params: Optional[HPIParams] = None,
    ) -> None:
        super().__init__(model_dir=model_dir, config=config)
        self._device = device or device_helper.get_default_device()
        self.batch_sampler.batch_size = batch_size
        self._onnx_format = use_onnx_model
        self._check_and_choose_model_format()
        self._hpi_params = hpi_params or {}
        self._hpi_config = self._get_hpi_config()
        self._ui_model = self.build_ui_model()
        self._data_reader = self._build_data_reader()

    def __call__(
        self,
        input: Any,
        batch_size: int = None,
        device: str = None,
        **kwargs: dict[str, Any],
    ) -> Iterator[Any]:
        self.set_predictor(batch_size, device)
        yield from self.apply(input, **kwargs)

    @property
    def model_path(self) -> Path:
        if self._onnx_format:
            return self.model_dir / f"{self.MODEL_FILE_PREFIX}.onnx"
        else:
            return self.model_dir / f"{self.MODEL_FILE_PREFIX}.pdmodel"

    @property
    def params_path(self) -> Union[Path, None]:
        if self._onnx_format:
            return None
        else:
            return self.model_dir / f"{self.MODEL_FILE_PREFIX}.pdiparams"

    def set_predictor(self, batch_size: int = None, device: str = None) -> None:
        if device and device != self._device:
            raise RuntimeError("Currently, changing devices is not supported.")
        if batch_size:
            self.batch_sampler.batch_size = batch_size

    def build_ui_model(self) -> BaseUltraInferModel:
        option = self._create_ui_option()
        return self._build_ui_model(option)

    def _get_hpi_config(self) -> HPIConfig:
        if HPI_CONFIG_KEY not in self.config:
            logging.debug("Key %r not found in the config", HPI_CONFIG_KEY)
        hpi_config = HPIConfig.model_validate(
            {
                **self.config.get(HPI_CONFIG_KEY, {}),
                **self._hpi_params.get("config", {}),
            }
        )
        return hpi_config

    def _create_ui_option(self) -> ui.RuntimeOption:
        option = ui.RuntimeOption()
        # HACK: Disable new IR for models that are known to have issues with the
        # new IR.
        if self.model_name in NEWIR_BLOCKLIST:
            option.paddle_infer_option.enable_new_ir = False
        device_type, device_ids = device_helper.parse_device(self._device)
        if device_type == "cpu":
            pass
        elif device_type == "gpu":
            if device_ids is None:
                device_ids = [0]
            if len(device_ids) > 1:
                logging.warning(
                    "Multiple devices are specified (%s), but only the first one will be used.",
                    self._device,
                )
            option.use_gpu(device_ids[0])
        else:
            assert_never(device_type)
        backend, backend_config = self._hpi_config.get_backend_and_config(
            model_name=self.model_name,
            device_type=device_type,
            onnx_format=self._onnx_format,
        )
        logging.info("Backend: %s", backend)
        logging.info("Backend config: %s", backend_config)
        backend_config.update_ui_option(option, self.model_dir)
        return option

    def _check_and_choose_model_format(self) -> None:
        has_onnx_model = any(self.model_dir.glob(f"{self.MODEL_FILE_PREFIX}.onnx"))
        has_pd_model = any(self.model_dir.glob(f"{self.MODEL_FILE_PREFIX}.pdmodel"))
        if self._onnx_format is None:
            if has_onnx_model and has_pd_model:
                logging.warning(
                    "Both ONNX and Paddle models are detected, but no preference is set. Default model (.pdmodel) will be used."
                )
            elif has_pd_model:
                logging.warning(
                    "Only Paddle model is detected. Paddle model will be used by default."
                )
            elif has_onnx_model:
                self._onnx_format = True
                logging.warning(
                    "Only ONNX model is detected. ONNX model will be used by default."
                )
            else:
                raise RuntimeError(
                    "No models are detected. Please ensure the model file exists."
                )
        elif self._onnx_format:
            if not has_onnx_model:
                raise RuntimeError(
                    "ONNX model is specified but not detected. Please ensure the ONNX model file exists."
                )
        else:
            if not has_pd_model:
                raise RuntimeError(
                    "Paddle model is specified but not detected. Please ensure the Paddle model file exists."
                )

    @abc.abstractmethod
    def _build_ui_model(self, option: ui.RuntimeOption) -> BaseUltraInferModel:
        raise NotImplementedError

    @abc.abstractmethod
    def _build_data_reader(self):
        raise NotImplementedError


class CVPredictor(HPPredictor):
    def _build_data_reader(self):
        return ReadImage(format="BGR")


class TSPredictor(HPPredictor):
    def _build_data_reader(self):
        return ReadTS()
