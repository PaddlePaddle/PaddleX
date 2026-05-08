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

from typing import Any, Dict, List, Tuple, Union

import numpy as np
from PIL import Image

from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import Normalize, ToBatch, ToCHWImage
from ..predictors import RunnerPredictor, TransformersPredictor
from .processors import DocTrPostProcess
from .result import DocTrResult

WARP_TRANSFORMERS_MODELS = ["UVDoc"]


class WarpRunnerPredictor(RunnerPredictor):
    """WarpRunnerPredictor that inherits from RunnerPredictor."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes WarpPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.preprocessors, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the warpping result, DocTrResult.

        Returns:
            type: The DocTrResult.
        """
        return DocTrResult

    def _build(self) -> Tuple:
        """Build the preprocessors and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="BGR")}
        preprocessors["Normalize"] = Normalize(mean=0.0, std=1.0, scale=1.0 / 255)
        preprocessors["ToCHW"] = ToCHWImage()
        preprocessors["ToBatch"] = ToBatch()
        postprocessors = {"DocTrPostProcess": DocTrPostProcess()}
        return preprocessors, postprocessors

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_raw_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.runner(x=x)
        batch_warp_preds = self.postprocessors["DocTrPostProcess"](batch_preds)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "doctr_img": batch_warp_preds,
        }


class WarpTransformersPredictor(TransformersPredictor):

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.read_op = ReadImage(format="BGR")
        self.image_processor, self.infer = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return DocTrResult

    def _build(self) -> Tuple:
        from transformers import AutoImageProcessor, AutoModel

        image_processor = self._load_pretrained_processor(AutoImageProcessor)
        model = self._load_pretrained_model(AutoModel)
        return image_processor, model

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        batch_raw_imgs = self.read_op(imgs=batch_data.instances)
        images = [Image.fromarray(img[..., ::-1]) for img in batch_raw_imgs]

        model_inputs = self.preprocess_images(images=images)
        original_images = model_inputs.pop("original_images")
        outputs = self.forward(model_inputs)
        batch_warp_preds = self.postprocess(outputs, original_images=original_images)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "doctr_img": batch_warp_preds,
        }

    def postprocess(self, outputs, *, original_images, **kwargs):
        results = self.image_processor.post_process_document_rectification(
            outputs.last_hidden_state, original_images=original_images
        )

        batch_warp_preds = []
        for res in results:
            warped = res["images"]
            warped = warped.detach().cpu().numpy()
            batch_warp_preds.append(warped)

        return batch_warp_preds
