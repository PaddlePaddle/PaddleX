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

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type

from ...common.batch_sampler import BaseBatchSampler
from ...utils.benchmark import add_inference_operations, benchmark
from .local_model_predictor import LocalModelPredictor

add_inference_operations("TransformersInfer")


class TransformersPredictor(LocalModelPredictor):
    """Base class for transformers-engine predictors."""

    __is_base = True

    def __init__(
        self,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            model_config=model_config,
            model_name=model_name,
            engine_config=engine_config,
            batch_size=batch_size,
            **kwargs,
        )

    def _require_model_dir(self) -> str:
        if self.model_dir is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `model_dir` for local inference."
            )
        return str(self.model_dir)

    def _build_pretrained_processor_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self._engine_config.get("tokenizer_kwargs") or {})
        kwargs.update(self._engine_config.get("processor_kwargs") or {})
        trust_remote_code = self._engine_config.get("trust_remote_code")
        if trust_remote_code is not None:
            kwargs.setdefault("trust_remote_code", trust_remote_code)
        return kwargs

    def _resolve_dtype(self, dtype: str):
        import torch

        if not hasattr(torch, dtype):
            raise ValueError(f"Unsupported torch dtype: {dtype!r}.")
        return getattr(torch, dtype)

    def _build_pretrained_model_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self._engine_config.get("model_kwargs") or {})
        trust_remote_code = self._engine_config.get("trust_remote_code")
        if trust_remote_code is not None:
            kwargs.setdefault("trust_remote_code", trust_remote_code)
        attn_implementation = self._engine_config.get("attn_implementation")
        if attn_implementation is not None:
            kwargs.setdefault("attn_implementation", attn_implementation)
        dtype = self._engine_config.get("dtype")
        if dtype is not None:
            kwargs.setdefault("dtype", self._resolve_dtype(dtype))
        return kwargs

    def _load_pretrained_processor(self, processor_cls: Type[Any]):
        kwargs = self._build_pretrained_processor_kwargs()
        return processor_cls.from_pretrained(self._require_model_dir(), **kwargs)

    def _load_pretrained_model(self, model_cls: Type[Any]):
        model = model_cls.from_pretrained(
            self._require_model_dir(), **self._build_pretrained_model_kwargs()
        )
        torch_device = self._get_manual_torch_device()
        if torch_device is not None:
            model = model.to(torch_device)
        model.eval()
        return model

    def _get_manual_torch_device(self) -> Optional[str]:
        device_type = self._engine_config.get("device_type")
        if device_type is None:
            return None
        device_id = self._engine_config.get("device_id")
        if device_type == "gpu":
            device_type = "cuda"
        if device_id is not None:
            return f"{device_type}:{device_id}"
        return device_type

    def _get_infer_device(self, model=None):
        import torch

        infer_model = model or getattr(self, "infer", None)
        if infer_model is None:
            return torch.device("cpu")
        if hasattr(infer_model, "device"):
            return infer_model.device
        try:
            return next(infer_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_to_infer_device(self, model_inputs, model=None):
        import torch

        infer_model = model or getattr(self, "infer", None)
        device = self._get_infer_device(model=infer_model)

        if hasattr(model_inputs, "to") and callable(getattr(model_inputs, "to")):
            model_inputs = model_inputs.to(device)
        elif isinstance(model_inputs, dict):
            model_inputs = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in model_inputs.items()
            }
        else:
            raise TypeError(
                f"Unsupported model_inputs type: {type(model_inputs)!r}; "
                "expected a Hugging Face BatchFeature/BatchEncoding or a dict of tensors."
            )

        target_dtype = None
        if infer_model is not None:
            try:
                target_dtype = next(infer_model.parameters()).dtype
            except StopIteration:
                pass

        if target_dtype is not None and target_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            for key in list(model_inputs.keys()):
                value = model_inputs[key]
                if torch.is_tensor(value) and value.is_floating_point():
                    if value.dtype != target_dtype:
                        model_inputs[key] = value.to(dtype=target_dtype)

        return model_inputs

    @abstractmethod
    def process(self, batch_data: List[Any]) -> Dict[str, List[Any]]:
        raise NotImplementedError

    @benchmark.timeit
    def preprocess_images(self, images, **kwargs):
        if getattr(self, "image_processor", None) is not None:
            processor = self.image_processor
        elif getattr(self, "processor", None) is not None:
            processor = self.processor
        else:
            raise ValueError(
                "Please make sure `self.image_processor` or `self.processor` is set before calling `preprocess_images`."
            )
        model_inputs = processor(images=images, return_tensors="pt", **kwargs)
        model_inputs = self._move_to_infer_device(model_inputs)

        return model_inputs

    @benchmark.timeit_with_options(name="TransformersInfer")
    def forward(self, model_inputs):
        if getattr(self, "infer", None) is None:
            raise ValueError(
                "Please make sure `self.infer` is set before calling `forward`."
            )

        import torch

        with torch.inference_mode():
            outputs = self.infer(**model_inputs)

        return outputs

    @benchmark.timeit_with_options(name="TransformersInfer")
    def generate(self, model_inputs, generate_kwargs):
        if getattr(self, "infer", None) is None:
            raise ValueError(
                "Please make sure `self.infer` is set before calling `forward`."
            )

        import torch

        with torch.inference_mode():
            outputs = self.infer.generate(**model_inputs, **generate_kwargs)

        return outputs

    @abstractmethod
    def postprocess(self, outputs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _build_batch_sampler(self) -> BaseBatchSampler:
        raise NotImplementedError

    @abstractmethod
    def _get_result_class(self) -> type:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "postprocess" in cls.__dict__:
            cls.postprocess = benchmark.timeit(cls.postprocess)
