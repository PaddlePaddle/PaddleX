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

"""Paddle dynamic graph runner."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from .....utils.deps import class_requires_deps
from .....utils.device import TemporaryDeviceChanger, constr_device


class PaddleDynamicRunnerConfig(BaseModel):
    """Engine config for paddle_dynamic inference."""

    model_config = ConfigDict(extra="forbid")

    device_type: Optional[str] = None
    device_id: Optional[int] = None


def _normalize_input(
    x: Union[Sequence[np.ndarray], np.ndarray, Dict[str, np.ndarray], None],
    kwargs: Dict[str, Any],
) -> Union[Sequence[np.ndarray], Dict[str, np.ndarray]]:
    """Normalize input from __call__(x=..., **kwargs) to a sequence or dict."""
    if x is None and "x" in kwargs:
        x = kwargs["x"]
    if x is None:
        raise TypeError("PaddleDynamicRunner.__call__ requires x")
    if isinstance(x, np.ndarray):
        return [x]
    return x


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _output_to_list(output: Any) -> List[np.ndarray]:
    """Convert model output to List[np.ndarray]."""
    if output is None:
        return []
    if isinstance(output, (list, tuple)):
        return [_to_numpy(v) for v in output]
    return [_to_numpy(output)]


@class_requires_deps("paddlepaddle")
class PaddleDynamicRunner:
    """InferenceRunner that wraps a Paddle dynamic graph model."""

    def __init__(
        self,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        infer_fn: Optional[Callable[[Any, Any], Any]] = None,
    ) -> None:
        """Wrap a dynamic model as an InferenceRunner.

        Args:
            model: The loaded model (eval mode).
            config: Engine config dict with device_type, device_id (like PaddleStaticRunner).
                Device is derived from config for inference.
            infer_fn: Optional custom inference function (model, x) -> output.
                Default: for Sequence (list/tuple) passes model(x); for dict passes model(**x).
        """
        self._model = model
        self._infer_fn = infer_fn
        cfg = config or {}
        device_type = cfg.get("device_type")
        if device_type:
            device_id = cfg.get("device_id")
            device_ids = [device_id] if device_id is not None else None
            self._device = constr_device(device_type, device_ids)
        else:
            self._device = None

    def __call__(
        self,
        x: Union[Sequence[np.ndarray], np.ndarray, Dict[str, np.ndarray], None] = None,
        **kwargs: Any,
    ) -> List[np.ndarray]:
        inputs = _normalize_input(x, kwargs)
        with TemporaryDeviceChanger(self._device):
            if self._infer_fn is not None:
                out = self._infer_fn(self._model, inputs)
            elif isinstance(inputs, dict):
                out = self._model(**inputs)
            else:
                out = self._model(inputs)
        return _output_to_list(out)
