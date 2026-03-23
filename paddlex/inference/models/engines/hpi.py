#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""HPI engine."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from pydantic import ValidationError

from ....constants import MODEL_FILE_PREFIX
from ....utils.deps import is_dep_available
from ....utils.device import get_default_device, parse_device
from ..bindings import Binding
from ..hpi import HPIInfo
from ..runners.hpi import HPIConfig, HPIRunner
from ..runners.inference_runner import InferenceRunner
from ..utils.model_paths import LocalModelFormat
from ._base import RunnerEngine


def _get_hpi_info(model_config: Optional[Dict[str, Any]]) -> Optional[HPIInfo]:
    if not model_config or "Hpi" not in model_config:
        return None
    try:
        return HPIInfo.model_validate(model_config["Hpi"])
    except ValidationError as e:
        raise RuntimeError(f"Invalid HPI info: {str(e)}") from e


class HPIEngine(RunnerEngine):
    """Engine for HPI / UltraInfer inference."""

    entities = "hpi"

    @property
    def name(self) -> str:
        return "hpi"

    @property
    def engine_config_model(self) -> Type[HPIConfig]:
        return HPIConfig

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        return ("paddle", "onnx", "om")

    def prepare_config_dict(
        self,
        raw: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw.setdefault("model_name", model_name or "")
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None
        elif "device_type" not in raw:
            raw["device_type"], _ = parse_device(get_default_device())
        return raw

    def get_config_dump_kwargs(self) -> Dict[str, Any]:
        return {"exclude_none": True, "by_alias": True}

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config
        if not is_dep_available("ultra-infer"):
            raise RuntimeError(
                "Engine 'hpi' is unavailable because dependency "
                "'ultra-infer' is not installed."
            )

    def build_runner(
        self,
        *,
        model_name: str,
        model_dir: Optional[Path],
        model_config: Optional[Dict[str, Any]],
        engine_config: Dict[str, Any],
        binding: Optional[Binding] = None,
    ) -> InferenceRunner:
        del binding
        if model_dir is None:
            raise ValueError("`model_dir` is required for engine='hpi'.")
        hpi_cfg = dict(engine_config)
        hpi_cfg.setdefault("model_name", model_name)
        if "hpi_info" not in hpi_cfg:
            hpi_info = _get_hpi_info(model_config)
            if hpi_info is not None:
                hpi_cfg["hpi_info"] = hpi_info
        hpi_config = HPIConfig.model_validate(hpi_cfg)
        return HPIRunner(
            model_dir=model_dir,
            model_file_prefix=MODEL_FILE_PREFIX,
            config=hpi_config,
        )
