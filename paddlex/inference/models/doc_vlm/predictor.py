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

import base64
import copy
import io
import os
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from ....utils import logging
from ....utils.deps import require_genai_client_plugin
from ....utils.device import TemporaryDeviceChanger
from ....utils.import_guard import import_paddle
from ...common.batch_sampler import DocVLMBatchSampler
from ...utils.misc import is_bfloat16_available
from ..predictors import (
    GenAIClientPredictor,
    LocalModelPredictor,
    TransformersPredictor,
)
from ..utils.model_paths import get_model_paths
from .constants import (
    PADDLEOCR_VL_GENAI_CLIENT_BATCH_SIZE,
    PADDLEOCR_VL_LOCAL_BATCH_SIZE,
    PADDLEOCR_VL_MAX_NEW_TOKENS,
)
from .result import DocVLMResult
from .utils import format_doc_vlm_result_dict, is_in_group


class DocVLMLocalPredictor(LocalModelPredictor):
    """DocVLM predictor for local model inference (Paddle dynamic graph)."""

    def __init__(self, *args, **kwargs):
        """Initializes DocVLMPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        bs = kwargs.get("batch_size", -1)
        if bs == -1:
            self.batch_sampler.batch_size = self._determine_batch_size()
        else:
            self.batch_sampler.batch_size = bs

        if is_bfloat16_available(self.device):
            self.dtype = "bfloat16"
        else:
            self.dtype = "float32"

        self.infer, self.processor = self._build(**kwargs)

        if (
            is_in_group(self.model_name, "PaddleOCR-VL")
            and self.batch_sampler.batch_size > PADDLEOCR_VL_LOCAL_BATCH_SIZE
        ):
            logging.warning(
                f"Currently, the {repr(self.model_name)} local model only supports batch size of {PADDLEOCR_VL_LOCAL_BATCH_SIZE}. "
                "The batch size will be updated."
            )
            self.batch_sampler.batch_size = PADDLEOCR_VL_LOCAL_BATCH_SIZE

    def _build_batch_sampler(self):
        """Builds and returns an DocVLMBatchSampler instance.

        Returns:
            DocVLMBatchSampler: An instance of DocVLMBatchSampler.
        """
        return DocVLMBatchSampler(self.model_name)

    def _get_result_class(self):
        """Returns the result class, DocVLMResult.

        Returns:
            type: The DocVLMResult class.
        """
        return DocVLMResult

    def _build(self, **kwargs):
        """Build the model, and correspounding processor on the configuration.

        Returns:
            model: An instance of Paddle model, could be either a dynamic model or a static model.
            processor: The correspounding processor for the model.
        """
        from .modeling import (
            PaddleOCRVLForConditionalGeneration,
            PPChart2TableInference,
            PPDocBee2Inference,
            PPDocBeeInference,
        )

        # build processor
        processor = self.build_processor()

        # build model
        if is_in_group(self.model_name, "PP-DocBee"):
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-DocBee series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PPDocBeeInference.from_pretrained(
                    self.model_dir, dtype=self.dtype
                )
        elif is_in_group(self.model_name, "PP-Chart2Table"):
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-Chart2Table series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model_path = get_model_paths(self.model_dir)

                if "safetensors" in model_path:
                    model = PPChart2TableInference.from_pretrained(
                        self.model_dir,
                        dtype=self.dtype,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_safetensors=True,
                        convert_from_hf=True,
                    )
                else:
                    model = PPChart2TableInference.from_pretrained(
                        self.model_dir,
                        dtype=self.dtype,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

        elif is_in_group(self.model_name, "PP-DocBee2"):
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-Chart2Table series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PPDocBee2Inference.from_pretrained(
                    self.model_dir,
                    dtype=self.dtype,
                )
        elif is_in_group(self.model_name, "PaddleOCR-VL"):
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PaddleOCR-VL series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                    self.model_dir,
                    dtype=self.dtype,
                    convert_from_hf=True,
                )
        else:
            raise NotImplementedError(f"Model {self.model_name} is not supported.")

        return model, processor

    def _determine_batch_size(self):
        if is_in_group(self.model_name, "PaddleOCR-VL"):
            batch_size = PADDLEOCR_VL_LOCAL_BATCH_SIZE
            logging.debug(
                f"The batch size of {self.model_name} is determined to be {batch_size}."
            )
            return batch_size
        else:
            raise RuntimeError(f"Could not determine batch size for {self.model_name}")

    def process(
        self,
        data: List[dict],
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            data (List[dict]): A batch of input data, must be a dict (e.g. {"image": /path/to/image, "query": some question}).

        Returns:
            dict: A dictionary containing the raw sample information and prediction results for every instance of the batch.
        """
        # FIXME: When `skip_special_tokens` is `True`, the results from different backends may differ.

        assert all(isinstance(i, dict) for i in data)

        src_data = copy.copy(data)
        # preprocess
        if is_in_group(self.model_name, "PaddleOCR-VL"):
            data = self.processor.preprocess(
                data, min_pixels=min_pixels, max_pixels=max_pixels
            )
        else:
            data = self.processor.preprocess(data)
            if min_pixels is not None:
                warnings.warn(
                    f"`min_pixels` is currently not supported by the {repr(self.model_name)} model and will be ignored."
                )
            if max_pixels is not None:
                warnings.warn(
                    f"`max_pixels` is currently not supported by the {repr(self.model_name)} model and will be ignored."
                )

        data = self._switch_inputs_to_device(data)

        # do infer
        generate_kwargs = {}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens
        elif is_in_group(self.model_name, "PaddleOCR-VL"):
            generate_kwargs["max_new_tokens"] = PADDLEOCR_VL_MAX_NEW_TOKENS
        if repetition_penalty is not None:
            warnings.warn(
                "`repetition_penalty` is currently not supported by the local model and will be ignored."
            )
        if temperature is not None:
            warnings.warn(
                "`temperature` is currently not supported by the local model and will be ignored."
            )
        if top_p is not None:
            warnings.warn(
                "`top_p` is currently not supported by the local model and will be ignored."
            )
        if use_cache is not None:
            generate_kwargs["use_cache"] = use_cache
        with TemporaryDeviceChanger(self.device):
            preds = self.infer.generate(
                data,
                **generate_kwargs,
            )

        # postprocess
        postprocess_kwargs = {}
        if skip_special_tokens is not None:
            postprocess_kwargs["skip_special_tokens"] = skip_special_tokens
        preds = self.processor.postprocess(preds, **postprocess_kwargs)

        result_dict = self._format_result_dict(preds, src_data)
        return result_dict

    def build_processor(self, **kwargs):
        from ..common.tokenizer import (
            LlamaTokenizer,
            MIXQwen2_5_Tokenizer,
            MIXQwen2Tokenizer,
            QWenTokenizer,
        )
        from ..common.tokenizer.tokenizer_utils import ChatTemplate
        from .processors import (
            GOTImageProcessor,
            PaddleOCRVLProcessor,
            PPChart2TableProcessor,
            PPDocBee2Processor,
            PPDocBeeProcessor,
            Qwen2_5_VLImageProcessor,
            Qwen2VLImageProcessor,
            SiglipImageProcessor,
        )

        if is_in_group(self.model_name, "PP-DocBee"):
            image_processor = Qwen2VLImageProcessor()
            tokenizer = MIXQwen2Tokenizer.from_pretrained(self.model_dir)
            return PPDocBeeProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )
        elif is_in_group(self.model_name, "PP-Chart2Table"):
            image_processor = GOTImageProcessor(1024)
            # Load GOT-OCR2 special tokens (<img>, </img>, <imgpad>, etc.)
            # from added_tokens.json so they encode as single tokens.
            extra_special_tokens = None
            added_tokens_file = Path(self.model_dir) / "added_tokens.json"
            if added_tokens_file.exists():
                import json

                with open(added_tokens_file) as f:
                    extra_special_tokens = json.load(f)
            tokenizer = QWenTokenizer(
                vocab_file=str(Path(self.model_dir) / "qwen.tiktoken"),
                extra_special_tokens=extra_special_tokens,
            )
            return PPChart2TableProcessor(
                image_processor=image_processor, tokenizer=tokenizer, dtype=self.dtype
            )
        elif is_in_group(self.model_name, "PP-DocBee2"):
            image_processor = Qwen2_5_VLImageProcessor()
            tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(self.model_dir)
            return PPDocBee2Processor(
                image_processor=image_processor, tokenizer=tokenizer
            )
        elif is_in_group(self.model_name, "PaddleOCR-VL"):
            image_processor = SiglipImageProcessor.from_pretrained(self.model_dir)
            vocab_file = str(Path(self.model_dir, "tokenizer.model"))
            tokenizer = LlamaTokenizer.from_pretrained(
                self.model_dir, vocab_file=vocab_file
            )
            # HACK
            chat_template_file = Path(self.model_dir, "chat_template.jinja")
            tokenizer.chat_template = ChatTemplate._compile_jinja_template(
                chat_template_file.read_text(encoding="utf-8")
            )
            return PaddleOCRVLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
            )
        else:
            raise NotImplementedError

    def _format_result_dict(self, model_preds, src_data):
        return format_doc_vlm_result_dict(model_preds, src_data, add_input_path=True)

    def _infer_dynamic_forward_device(self, device):
        """infer the forward device for dynamic graph model"""
        import GPUtil

        from ....utils.device import parse_device

        if device is None:
            return None
        if "cpu" in device.lower():
            return "cpu"
        device_type, device_ids = parse_device(device)

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices is None:
            env_gpu_num = len(GPUtil.getGPUs())
            cuda_visible_devices = ",".join([str(i) for i in range(env_gpu_num)])
        env_device_ids = cuda_visible_devices.split(",")
        for env_device_id in env_device_ids:
            if not env_device_id.isdigit():
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES ID must be an integer. Invalid device ID: {env_device_id}"
                )

        if max(device_ids) >= len(env_device_ids):
            raise ValueError(
                f"Required gpu ids {device_ids} even larger than the number of visible devices {cuda_visible_devices}."
            )

        rst_global_gpu_ids = [env_device_ids[idx] for idx in device_ids]
        return device_type + ":" + ",".join(rst_global_gpu_ids)

    def _switch_inputs_to_device(self, input_dict):
        """Switch the input to the specified device"""
        paddle = import_paddle()

        if self.device is None:
            return input_dict
        rst_dict = {
            k: (
                paddle.to_tensor(input_dict[k], place=self.device)
                if isinstance(input_dict[k], paddle.Tensor)
                else input_dict[k]
            )
            for k in input_dict
        }
        return rst_dict


class DocVLMGenAIClientPredictor(GenAIClientPredictor):
    """DocVLM predictor for remote GenAI inference via GenAIClient."""

    def __init__(self, *args, **kwargs):
        engine_config = kwargs.pop("engine_config", None)
        model_name = kwargs.pop("model_name", "")
        if engine_config is None or not engine_config:
            raise ValueError("DocVLMGenAIClientPredictor requires `engine_config`.")
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
        )
        bs = kwargs.get("batch_size", 1)
        if bs == -1 and is_in_group(self.model_name, "PaddleOCR-VL"):
            bs = PADDLEOCR_VL_GENAI_CLIENT_BATCH_SIZE
        elif bs == -1:
            bs = 1
        self.batch_sampler.batch_size = bs

    def _build_batch_sampler(self):
        return DocVLMBatchSampler(self.model_name)

    def _get_result_class(self):
        return DocVLMResult

    def __call__(self, input, batch_size=None, **kwargs):
        yield from self.apply(input, **kwargs)

    def predict(self, input, **kwargs):
        """Alias for __call__ for pipeline compatibility."""
        yield from self(input, **kwargs)

    def apply(self, input, **kwargs):
        from .result import DocVLMResult

        for instances in self.batch_sampler(input):
            if not isinstance(instances, list):
                instances = [instances]
            pred = self.process(instances, **kwargs)
            for idx in range(len(pred.get("result", []))):
                single = {
                    k: (v[idx] if isinstance(v, list) else v) for k, v in pred.items()
                }
                yield DocVLMResult(single)

    def process(
        self,
        data: List[dict],
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        require_genai_client_plugin()
        preds = self._genai_client_process(
            data,
            max_new_tokens=max_new_tokens,
            skip_special_tokens=skip_special_tokens,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return format_doc_vlm_result_dict(preds, data, add_input_path=True)

    def _genai_client_process(
        self,
        data,
        max_new_tokens,
        skip_special_tokens,
        repetition_penalty,
        temperature,
        top_p,
        min_pixels,
        max_pixels,
    ):
        client = self.genai_client
        futures = []
        if client.backend == "llama-cpp-server":
            image_format = "PNG"
        else:
            image_format = "JPEG"
        try:
            for item in data:
                image = item["image"]
                if isinstance(image, str):
                    if image.startswith("http://") or image.startswith("https://"):
                        image_url = image
                    else:
                        from PIL import Image

                        with Image.open(image) as img:
                            img = img.convert("RGB")
                            with io.BytesIO() as buf:
                                img.save(buf, format=image_format)
                                image_url = (
                                    f"data:image/{image_format.lower()};base64,"
                                    + base64.b64encode(buf.getvalue()).decode("ascii")
                                )
                elif isinstance(image, np.ndarray):
                    import cv2
                    from PIL import Image

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(image)
                    with io.BytesIO() as buf:
                        img.save(buf, format=image_format)
                        image_url = (
                            f"data:image/{image_format.lower()};base64,"
                            + base64.b64encode(buf.getvalue()).decode("ascii")
                        )
                else:
                    raise TypeError(f"Not supported image type: {type(image)}")

                if client.backend == "fastdeploy-server":
                    kwargs = {
                        "temperature": 1 if temperature is None else temperature,
                        "top_p": 0 if top_p is None else top_p,
                    }
                else:
                    kwargs = {
                        "temperature": 0 if temperature is None else temperature,
                    }
                    if top_p is not None:
                        kwargs["top_p"] = top_p

                if client.backend in ["mlx-vlm-server", "llama-cpp-server"]:
                    max_tokens_name = "max_tokens"
                else:
                    max_tokens_name = "max_completion_tokens"

                if max_new_tokens is not None:
                    kwargs[max_tokens_name] = max_new_tokens
                elif is_in_group(self.model_name, "PaddleOCR-VL"):
                    kwargs[max_tokens_name] = PADDLEOCR_VL_MAX_NEW_TOKENS

                kwargs["extra_body"] = {}
                if skip_special_tokens is not None:
                    if client.backend in (
                        "fastdeploy-server",
                        "vllm-server",
                        "sglang-server",
                        "mlx-vlm-server",
                        "llama-cpp-server",
                    ):
                        kwargs["extra_body"][
                            "skip_special_tokens"
                        ] = skip_special_tokens
                    else:
                        raise ValueError("Not supported")

                if repetition_penalty is not None:
                    kwargs["extra_body"]["repetition_penalty"] = repetition_penalty

                if min_pixels is not None:
                    if client.backend == "vllm-server":
                        kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                            "extra_body"
                        ].get("mm_processor_kwargs", {})
                        kwargs["extra_body"]["mm_processor_kwargs"][
                            "min_pixels"
                        ] = min_pixels
                    else:
                        warnings.warn(
                            f"{repr(client.backend)} does not support `min_pixels`."
                        )

                if max_pixels is not None:
                    if client.backend == "vllm-server":
                        kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                            "extra_body"
                        ].get("mm_processor_kwargs", {})
                        kwargs["extra_body"]["mm_processor_kwargs"][
                            "max_pixels"
                        ] = max_pixels
                    else:
                        warnings.warn(
                            f"{repr(client.backend)} does not support `max_pixels`."
                        )

                future = client.create_chat_completion(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {"type": "text", "text": item["query"]},
                            ],
                        }
                    ],
                    return_future=True,
                    timeout=600,
                    **kwargs,
                )

                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                results.append(result.choices[0].message.content)

            return results
        except Exception:
            # Cancel all pending futures to avoid wasting resources
            for future in futures:
                if not future.done():
                    future.cancel()
            raise


class DocVLMTransformersPredictor(TransformersPredictor):
    """DocVLM predictor backed by Hugging Face transformers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.batch_sampler.batch_size == -1:
            self.batch_sampler.batch_size = PADDLEOCR_VL_LOCAL_BATCH_SIZE
        self.processor, self.infer = self._build()

    def _build_batch_sampler(self):
        return DocVLMBatchSampler(self.model_name)

    def _get_result_class(self):
        return DocVLMResult

    def _build(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = self._load_pretrained_processor(AutoProcessor)
        model = self._load_pretrained_model(AutoModelForImageTextToText)
        return processor, model

    def process(
        self,
        data: List[dict],
        max_new_tokens: Optional[int] = None,
        skip_special_tokens: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        from .processors.common import fetch_image

        assert all(isinstance(i, dict) for i in data)
        src_data = copy.copy(data)

        images = []
        texts = []
        for item in data:
            image = fetch_image(item["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": item.get("query", "")},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images.append(image)
            texts.append(prompt)

        if is_in_group(self.model_name, "PaddleOCR-VL"):
            images_kwargs = {"size": dict(self.processor.image_processor.size)}
            if min_pixels is not None:
                images_kwargs["size"]["shortest_edge"] = min_pixels
            if max_pixels is not None:
                images_kwargs["size"]["longest_edge"] = max_pixels
            model_inputs = self.preprocess_images(
                images=images, text=texts, images_kwargs=images_kwargs
            )
        else:
            model_inputs = self.preprocess_images(images=images, text=texts)

        generate_kwargs = {
            "max_new_tokens": (
                max_new_tokens
                if max_new_tokens is not None
                else PADDLEOCR_VL_MAX_NEW_TOKENS
            )
        }
        if repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p
        if use_cache is not None:
            generate_kwargs["use_cache"] = use_cache

        generated_ids = self.generate(model_inputs, generate_kwargs)

        preds = self.postprocess(
            generated_ids,
            model_inputs=model_inputs,
            skip_special_tokens=skip_special_tokens,
        )

        return format_doc_vlm_result_dict(preds, src_data, add_input_path=True)

    def postprocess(self, outputs, *, model_inputs, skip_special_tokens, **kwargs):
        prompt_ids = model_inputs["input_ids"]
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(prompt_ids, outputs)
        ]
        preds = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=(
                True if skip_special_tokens is None else skip_special_tokens
            ),
            clean_up_tokenization_spaces=False,
        )

        return preds
