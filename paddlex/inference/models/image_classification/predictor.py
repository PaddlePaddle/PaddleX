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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ....modules.image_classification.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import Normalize, Resize, ResizeByShort, ToBatch, ToCHWImage
from ..predictors import RunnerPredictor, TransformersPredictor
from .processors import Crop, Topk
from .result import TopkResult

PPLCNET_MODELS = [name for name in MODELS if name.startswith("PP-LCNet_")]
HGNETV2_MODELS = [name for name in MODELS if name.startswith("PP-HGNetV2")]
CLAS_TRANSFORMERS_MODELS = PPLCNET_MODELS + HGNETV2_MODELS


class ClasRunnerPredictor(RunnerPredictor):
    """ClasRunnerPredictor that inherits from RunnerPredictor."""

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self, topk: Union[int, None] = None, *args: List, **kwargs: Dict
    ) -> None:
        """Initializes ClasPredictor.

        Args:
            topk (int, optional): The number of top-k predictions to return. If None, it will be depending on config of inference or predict. Defaults to None.
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.preprocessors, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, TopkResult.

        Returns:
            type: The TopkResult class.
        """
        return TopkResult

    def _build(self) -> Tuple:
        """Build the preprocessors and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="RGB")}
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            preprocessors[name] = op
        preprocessors["ToBatch"] = ToBatch()

        postprocessors = {}
        for key in self.config["PostProcess"]:
            func = self._FUNC_MAP.get(key)
            args = self.config["PostProcess"].get(key, {})
            name, op = func(self, **args) if args else func(self)
            postprocessors[name] = op
        return preprocessors, postprocessors

    def process(
        self, batch_data: List[Union[str, np.ndarray]], topk: Union[int, None] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            topk: The number of top predictions to keep. If None, it will be depending on `self.topk`. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data.instances)
        batch_imgs = self.preprocessors["Resize"](imgs=batch_raw_imgs)
        if "Crop" in self.preprocessors:
            batch_imgs = self.preprocessors["Crop"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.runner(x=x)
        batch_class_ids, batch_scores, batch_label_names = self.postprocessors["Topk"](
            batch_preds, topk=topk or self.topk
        )
        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "class_ids": batch_class_ids,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    @register("ResizeImage")
    # TODO(gaotingquan): backend & interpolation
    def build_resize(
        self, resize_short=None, size=None, backend="cv2", interpolation="LINEAR"
    ):
        assert resize_short or size
        if resize_short:
            op = ResizeByShort(
                target_short_edge=resize_short,
                size_divisor=None,
                interp=interpolation,
                backend=backend,
            )
        else:
            op = Resize(
                target_size=size,
                size_divisor=None,
                interp=interpolation,
                backend=backend,
            )
        return "Resize", op

    @register("CropImage")
    def build_crop(self, size=224):
        return "Crop", Crop(crop_size=size)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="",
        channel_num=3,
    ):
        assert channel_num == 3
        assert order == ""
        return "Normalize", Normalize(scale=scale, mean=mean, std=std)

    @register("ToCHWImage")
    def build_to_chw(self):
        return "ToCHW", ToCHWImage()

    @register("Topk")
    def build_topk(self, topk, label_list=None):
        if not self.topk:
            self.topk = int(topk)
        return "Topk", Topk(class_ids=label_list)


class ClasTransformersPredictor(TransformersPredictor):
    """Image classification predictor backed by Hugging Face transformers."""

    def __init__(self, topk: Optional[int] = None, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.topk = topk
        self._load_default_topk()
        self.read_op = ReadImage(format="RGB")
        self.image_processor, self.infer = self._build()

    def _load_default_topk(self) -> None:
        if self.topk is not None:
            return
        post = self.model_config.get("PostProcess", {})
        topk_cfg = post.get("Topk", {})
        if isinstance(topk_cfg, dict) and topk_cfg.get("topk") is not None:
            self.topk = int(topk_cfg["topk"])
        else:
            self.topk = 5

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TopkResult

    def _build(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        image_processor = self._load_pretrained_processor(AutoImageProcessor)
        model = self._load_pretrained_model(AutoModelForImageClassification)
        return image_processor, model

    def _resolve_id2label(self) -> Dict[int, str]:
        raw = dict(self.infer.config.id2label or {})
        return {int(k): str(v) for k, v in raw.items()}

    def _resolve_logits(self, outputs):
        logits = getattr(outputs, "logits", None)
        if logits is not None:
            return logits

        # Some Paddle-converted HF classification models expose class scores as
        # `last_hidden_state` instead of `logits`.
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if (
            last_hidden_state is not None
            and getattr(last_hidden_state, "ndim", None) == 2
        ):
            return last_hidden_state

        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None and getattr(pooler_output, "ndim", None) == 2:
            return pooler_output

        if hasattr(outputs, "to_tuple"):
            for item in outputs.to_tuple():
                if getattr(item, "ndim", None) == 2:
                    return item

        raise AttributeError(
            f"{type(outputs).__name__!r} does not provide a usable classification logits tensor."
        )

    def process(
        self, batch_data: List[Union[str, np.ndarray]], topk: Optional[int] = None
    ) -> Dict[str, Any]:
        batch_raw_imgs = self.read_op(imgs=batch_data.instances)
        images = [Image.fromarray(img) for img in batch_raw_imgs]

        model_inputs = self.preprocess_images(images=images)
        outputs = self.forward(model_inputs)
        indexes, batch_scores, batch_label_names = self.postprocess(outputs, topk=topk)

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": batch_raw_imgs,
            "class_ids": indexes,
            "scores": batch_scores,
            "label_names": batch_label_names,
        }

    def postprocess(self, outputs, *, topk, **kwargs):
        import torch

        id2label = self._resolve_id2label()
        logits = self._resolve_logits(outputs)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        k = int(topk if topk is not None else self.topk)
        indexes = probs.argsort(axis=1)[:, -k:][:, ::-1].astype("int32")
        batch_scores = [
            np.around(probs[i][idx], decimals=5) for i, idx in enumerate(indexes)
        ]
        batch_label_names = [
            [id2label.get(int(i), str(i)) for i in row] for row in indexes
        ]

        return indexes, batch_scores, batch_label_names
