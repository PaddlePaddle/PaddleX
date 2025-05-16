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

import copy
import os
import warnings
from typing import List

from ....modules.doc_vlm.model_list import MODELS
from ....utils.device import TemporaryDeviceChanger
from ....utils.env import get_device_type
from ...common.batch_sampler import DocVLMBatchSampler
from ..base import BasePredictor
from .result import DocVLMResult


class DocVLMPredictor(BasePredictor):

    entities = MODELS
    model_group = {
        "PP-DocBee": {"PP-DocBee-2B", "PP-DocBee-7B"},
        "PP-DocBee2": {"PP-DocBee2-3B"},
        "PP-Chart2Table": {"PP-Chart2Table"},
    }

    def __init__(self, *args, **kwargs):
        """Initializes DocVLMPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        import paddle

        super().__init__(*args, **kwargs)
        self.device = kwargs.get("device", None)
        self.dtype = (
            "bfloat16"
            if ("npu" in get_device_type() or paddle.amp.is_bfloat16_supported())
            and (self.device is None or "cpu" not in self.device)
            else "float32"
        )

        self.infer, self.processor = self._build(**kwargs)

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
        else:
            raise NotImplementedError(f"Model {self.model_name} is not supported.")

        return model, processor

    def process(self, data: List[dict], **kwargs):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            data (List[dict]): A batch of input data, must be a dict (e.g. {"image": /path/to/image, "query": some question}).
            kwargs (Optional[dict]): Arbitrary keyword arguments passed to model.generate.

        Returns:
            dict: A dictionary containing the raw sample information and prediction results for every instance of the batch.
        """
        assert all(isinstance(i, dict) for i in data)

        src_data = copy.copy(data)
        # preprocess
        data = self.processor.preprocess(data)
        data = self._switch_inputs_to_device(data)

        # do infer
        with TemporaryDeviceChanger(self.device):
            preds = self.infer.generate(data, **kwargs)

        # postprocess
        preds = self.processor.postprocess(preds)

        result_dict = self._format_result_dict(preds, src_data)
        return result_dict

    def build_processor(self, **kwargs):
        from ..common.tokenizer import (
            MIXQwen2_5_Tokenizer,
            MIXQwen2Tokenizer,
            QWenTokenizer,
        )
        from .processors import (
            GOTImageProcessor,
            PPChart2TableProcessor,
            PPDocBee2Processor,
            PPDocBeeProcessor,
            Qwen2_5_VLImageProcessor,
            Qwen2VLImageProcessor,
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
        else:
            raise NotImplementedError

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
