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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np

from ....modules.doc_vlm.model_list import MODELS
from ....utils import logging
from ....utils.deps import require_genai_client_plugin
from ....utils.device import TemporaryDeviceChanger
from ...common.batch_sampler import DocVLMBatchSampler
from ...utils.misc import is_bfloat16_available, is_float16_available
from ..base import BasePredictor
from .result import DocVLMResult


class DocVLMPredictor(BasePredictor):

    entities = MODELS
    model_group = {
        "PP-DocBee": {"PP-DocBee-2B", "PP-DocBee-7B"},
        "PP-DocBee2": {"PP-DocBee2-3B"},
        "PP-Chart2Table": {"PP-Chart2Table"},
        "PaddleOCR-VL": {"PaddleOCR-VL-0.9B"},
    }

    def __init__(self, *args, **kwargs):
        """Initializes DocVLMPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        if self._use_local_model:
            self.device = kwargs.get("device", None)
            if is_bfloat16_available(self.device):
                self.dtype = "bfloat16"
            elif is_float16_available(self.device):
                self.dtype = "float16"
            else:
                self.dtype = "float32"

            self.infer, self.processor = self._build(**kwargs)

            if (
                self.model_name == "PaddleOCR-VL-0.9B"
                and self.batch_sampler.batch_size > 1
            ):
                logging.warning(
                    "Currently, the PaddleOCR-VL-0.9B local model only supports batch size of 1. The batch size will be updated to 1."
                )
                self.batch_sampler.batch_size = 1
        else:
            if self.batch_sampler.batch_size > 1:
                self._thread_pool = ThreadPoolExecutor(
                    max_workers=min(self.batch_sampler.batch_size, os.cpu_count() or 1)
                )

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
        if self.model_name in self.model_group["PP-DocBee"]:
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-DocBee series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PPDocBeeInference.from_pretrained(
                    self.model_dir, dtype=self.dtype
                )
        elif self.model_name in self.model_group["PP-Chart2Table"]:
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-Chart2Table series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PPChart2TableInference.from_pretrained(
                    self.model_dir,
                    dtype=self.dtype,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
        elif self.model_name in self.model_group["PP-DocBee2"]:
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PP-Chart2Table series does not support `use_hpip=True` for now."
                )
            with TemporaryDeviceChanger(self.device):
                model = PPDocBee2Inference.from_pretrained(
                    self.model_dir,
                    dtype=self.dtype,
                )
        elif self.model_name in self.model_group["PaddleOCR-VL"]:
            if kwargs.get("use_hpip", False):
                warnings.warn(
                    "The PaddelOCR-VL series does not support `use_hpip=True` for now."
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
        # TODO: Sampling settings
        # FIXME: When `skip_special_tokens` is `True`, the results from different backends may differ.

        assert all(isinstance(i, dict) for i in data)

        if self._use_local_model:
            src_data = copy.copy(data)
            # preprocess
            data = self.processor.preprocess(data)
            data = self._switch_inputs_to_device(data)

            # do infer
            generate_kwargs = {}
            if max_new_tokens is not None:
                generate_kwargs["max_new_tokens"] = max_new_tokens
            elif self.model_name in self.model_group["PaddleOCR-VL"]:
                generate_kwargs["max_new_tokens"] = 8192
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
            if min_pixels is not None:
                warnings.warn(
                    "`min_pixels` is currently not supported by the local model and will be ignored."
                )
            if max_pixels is not None:
                warnings.warn(
                    "`max_pixels` is currently not supported by the local model and will be ignored."
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
        else:
            require_genai_client_plugin()

            src_data = data

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

        if self.model_name in self.model_group["PP-DocBee"]:
            image_processor = Qwen2VLImageProcessor()
            tokenizer = MIXQwen2Tokenizer.from_pretrained(self.model_dir)
            return PPDocBeeProcessor(
                image_processor=image_processor, tokenizer=tokenizer
            )
        elif self.model_name in self.model_group["PP-Chart2Table"]:
            image_processor = GOTImageProcessor(1024)
            tokenizer = QWenTokenizer.from_pretrained(self.model_dir)
            return PPChart2TableProcessor(
                image_processor=image_processor, tokenizer=tokenizer, dtype=self.dtype
            )
        elif self.model_name in self.model_group["PP-DocBee2"]:
            image_processor = Qwen2_5_VLImageProcessor()
            tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(self.model_dir)
            return PPDocBee2Processor(
                image_processor=image_processor, tokenizer=tokenizer
            )
        elif self.model_name in self.model_group["PaddleOCR-VL"]:
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

    def close(self):
        super().close()
        if hasattr(self, "_thread_pool"):
            self._thread_pool.shutdown()

    def _format_result_dict(self, model_preds, src_data):
        if not isinstance(model_preds, list):
            model_preds = [model_preds]
        if not isinstance(src_data, list):
            src_data = [src_data]
        if len(model_preds) != len(src_data):
            raise ValueError(
                f"Model predicts {len(model_preds)} results while src data has {len(src_data)} samples."
            )

        rst_format_dict = {k: [] for k in src_data[0].keys()}
        rst_format_dict["result"] = []

        for data_sample, model_pred in zip(src_data, model_preds):
            for k in data_sample.keys():
                rst_format_dict[k].append(data_sample[k])
            rst_format_dict["result"].append(model_pred)

        return rst_format_dict

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
        import paddle

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
        lock = Lock()

        def _process(item):
            image = item["image"]
            if isinstance(image, str):
                if image.startswith("http://") or image.startswith("https://"):
                    image_url = image
                else:
                    from PIL import Image

                    with Image.open(image) as img:
                        img = img.convert("RGB")
                        with io.BytesIO() as buf:
                            img.save(buf, format="JPEG")
                            image_url = "data:image/jpeg;base64," + base64.b64encode(
                                buf.getvalue()
                            ).decode("ascii")
            elif isinstance(image, np.ndarray):
                import cv2
                from PIL import Image

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
                with io.BytesIO() as buf:
                    img.save(buf, format="JPEG")
                    image_url = "data:image/jpeg;base64," + base64.b64encode(
                        buf.getvalue()
                    ).decode("ascii")
            else:
                raise TypeError(f"Not supported image type: {type(image)}")

            if self._genai_client.backend == "fastdeploy-server":
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

            if max_new_tokens is not None:
                kwargs["max_completion_tokens"] = max_new_tokens
            elif self.model_name in self.model_group["PaddleOCR-VL"]:
                kwargs["max_completion_tokens"] = 8192

            kwargs["extra_body"] = {}
            if skip_special_tokens is not None:
                if self._genai_client.backend in (
                    "fastdeploy-server",
                    "vllm-server",
                    "sglang-server",
                ):
                    kwargs["extra_body"]["skip_special_tokens"] = skip_special_tokens
                else:
                    raise ValueError("Not supported")

            if repetition_penalty is not None:
                kwargs["extra_body"]["repetition_penalty"] = repetition_penalty

            if min_pixels is not None:
                if self._genai_client.backend == "vllm-server":
                    kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                        "extra_body"
                    ].get("mm_processor_kwargs", {})
                    kwargs["extra_body"]["mm_processor_kwargs"][
                        "min_pixels"
                    ] = min_pixels
                else:
                    warnings.warn(
                        f"{repr(self._genai_client.backend)} does not support `min_pixels`."
                    )

            if max_pixels is not None:
                if self._genai_client.backend == "vllm-server":
                    kwargs["extra_body"]["mm_processor_kwargs"] = kwargs[
                        "extra_body"
                    ].get("mm_processor_kwargs", {})
                    kwargs["extra_body"]["mm_processor_kwargs"][
                        "max_pixels"
                    ] = max_pixels
                else:
                    warnings.warn(
                        f"{repr(self._genai_client.backend)} does not support `max_pixels`."
                    )

            with lock:
                future = self._genai_client.create_chat_completion(
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
                return future

        if len(data) > 1:
            futures = list(self._thread_pool.map(_process, data))
        else:
            futures = [_process(data[0])]

        results = []
        for future in futures:
            result = future.result()
            results.append(result.choices[0].message.content)

        return results
