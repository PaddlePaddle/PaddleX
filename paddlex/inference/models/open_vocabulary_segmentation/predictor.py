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


from typing import Any, Dict, List

from ....modules.open_vocabulary_segmentation.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..base import BasePredictor
from .processors import SAMProcessor
from .results import SAMSegResult


class OVSegPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args, **kwargs):
        """Initializes DetPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.pre_ops, self.infer, self.processor = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return SAMSegResult

    def _build(self):
        # build model preprocess ops
        pre_ops = [ReadImage(format="RGB")]
        for cfg in self.config.get("Preprocess", []):
            tf_key = cfg["type"]
            func = self._FUNC_MAP[tf_key]
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            if op:
                pre_ops.append(op)

        # build infer
        infer = self.create_static_infer()

        # build model specific processor, it's required for a OV model.
        processor_cfg = self.config["Processor"]
        tf_key = processor_cfg["type"]
        func = self._FUNC_MAP[tf_key]
        processor_cfg.pop("type")
        args = processor_cfg
        processor = func(self, **args) if args else func(self)

        return pre_ops, infer, processor

    def process(self, batch_data: List[Any], prompts: Dict[str, Any]):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[str]): A batch of input data (e.g., image file paths).
            prompt (Dict[str, Any]): Prompt for open vocabulary segmentation.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        image_paths = batch_data.input_paths
        src_images = self.pre_ops[0](batch_data.instances)
        datas = src_images
        # preprocess
        for pre_op in self.pre_ops[1:-1]:
            datas = pre_op(datas)

        # use Model-specific preprocessor to format batch inputs
        batch_inputs = self.processor.preprocess(datas, **prompts)

        # do infer
        batch_preds = self.infer(batch_inputs)

        # postprocess
        masks = self.processor.postprocess(batch_preds)

        return {
            "input_path": image_paths,
            "input_img": src_images,
            "prompts": [prompts] * len(image_paths),
            "masks": masks,
        }

    @register("SAMProcessor")
    def build_sam_preprocessor(
        self, size=1024, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
    ):
        return SAMProcessor(size=size, img_mean=mean, img_std=std)
