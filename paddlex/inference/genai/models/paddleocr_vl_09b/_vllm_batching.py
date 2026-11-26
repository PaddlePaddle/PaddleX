# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Multi-image vision batching for vLLM's per-image PaddleOCR-VL input schema.

Adapted from the Apache-2.0 PaddleOCR-VL implementations in PaddleX and vLLM.
The processor, encoder, attention backend, projector and decoder stay native.
"""

import inspect
import math
from itertools import accumulate

from .....utils import logging
from .....utils.deps import is_dep_available

if not all(map(is_dep_available, ("torch", "vllm"))):
    raise ImportError("PaddleOCR-VL batching requires Torch and vLLM.")
else:
    import torch
    from vllm.model_executor.models.paddleocr_vl import (
        PaddleOCRVLDummyInputsBuilder,
        PaddleOCRVLForConditionalGeneration,
        PaddleOCRVLMultiModalProcessor,
        PaddleOCRVLProcessingInfo,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    PaddleOCRVLMultiModalProcessor,
    info=PaddleOCRVLProcessingInfo,
    dummy_inputs=PaddleOCRVLDummyInputsBuilder,
)
class PaddleXVisionBatchedPaddleOCRVLForConditionalGeneration(
    PaddleOCRVLForConditionalGeneration
):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        encoder = getattr(getattr(self.visual, "vision_model", None), "encoder", None)
        backend = getattr(getattr(encoder, "attn_backend", None), "name", None)
        try:
            params = inspect.signature(self.visual.forward).parameters
        except (TypeError, ValueError):
            params = {}
        self._paddlex_batch_images = (
            backend in ("FLASH_ATTN", "TORCH_SDPA")
            and self.visual.dtype in (torch.float16, torch.bfloat16)
            and vllm_config.quant_config is None
            and vllm_config.parallel_config.tensor_parallel_size == 1
            and all(
                name in params
                and params[name].kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for name in (
                    "pixel_values",
                    "image_grid_thw",
                    "position_ids",
                    "interpolate_pos_encoding",
                    "cu_seqlens",
                )
            )
        )
        if self._paddlex_batch_images:
            logging.info(f"PaddleX multi-image vision batching enabled ({backend}).")
        else:
            logging.warning(
                f"PaddleX multi-image batching is unsupported with this vision setup ({backend}); "
                "using vLLM's native image path."
            )

    def _process_image_input(self, image_input):
        grid_tensor = image_input.image_grid_thw
        if (
            not self._paddlex_batch_images
            or not isinstance(grid_tensor, torch.Tensor)
            or grid_tensor.ndim != 2
            or grid_tensor.shape[1] != 3
            or grid_tensor.shape[0] < 2
            or grid_tensor.is_floating_point()
            or grid_tensor.is_complex()
            or grid_tensor.dtype == torch.bool
        ):
            return super()._process_image_input(image_input)

        grids = [tuple(grid) for grid in grid_tensor.tolist()]
        lengths = [math.prod(grid) for grid in grids]
        pixels = image_input.pixel_values
        if isinstance(pixels, torch.Tensor) and pixels.ndim == 5:
            pixels = pixels.unbind()
        if (
            not isinstance(pixels, (tuple, list))
            or len(pixels) != len(lengths)
            or not isinstance(pixels[0], torch.Tensor)
            or sum(lengths) >= 2**31
            or any(min(grid) <= 0 for grid in grids)
            or any(
                not isinstance(image, torch.Tensor)
                or image.ndim != 4
                or image.shape[0] != n
                or image.shape[1:] != pixels[0].shape[1:]
                or image.device != pixels[0].device
                or image.dtype != pixels[0].dtype
                for image, n in zip(pixels, lengths)
            )
        ):
            return super()._process_image_input(image_input)

        # One segment per image. Keep projection and output ordering native.
        pixels = torch.cat(pixels).to(dtype=self.visual.dtype)
        positions = torch.cat(
            [
                torch.arange(n, device="cpu") % (h * w)
                for n, (_, h, w) in zip(lengths, grids)
            ]
        ).to(pixels.device, non_blocking=True)
        boundaries = torch.tensor(
            [0, *accumulate(lengths)], dtype=torch.int32, device="cpu"
        )
        encoded = self.visual(
            pixel_values=pixels,
            image_grid_thw=grids,
            position_ids=positions,
            interpolate_pos_encoding=True,
            cu_seqlens=boundaries,
        ).squeeze(0)
        return self.mlp_AR(encoded.split(lengths), grid_tensor)
