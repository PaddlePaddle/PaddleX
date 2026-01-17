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

from typing import Any, List, Optional, Tuple, Union

from ....modules.object_detection.model_list import LAYOUTANALYSIS_MODELS
from ..object_detection import DetPredictor
from ..object_detection.processors import Resize, ToBatch
from .processors import LayoutAnalysisProcess
from .result import LayoutAnalysisResult
from .utils import STATIC_SHAPE_MODEL_LIST


class LayoutAnalysisPredictor(DetPredictor):

    entities = LAYOUTANALYSIS_MODELS

    def __init__(
        self,
        *args,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """Initializes LayoutAnalysisPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            img_size (Optional[Union[int, Tuple[int, int]]], optional): The input image size (w, h). Defaults to None.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        if img_size is not None:
            assert (
                self.model_name not in STATIC_SHAPE_MODEL_LIST
            ), f"The model {self.model_name} is not supported set input shape"
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            elif isinstance(img_size, (tuple, list)):
                assert len(img_size) == 2, f"The length of `img_size` should be 2."
            else:
                raise ValueError(
                    f"The type of `img_size` must be int or Tuple[int, int], but got {type(img_size)}."
                )
        super().__init__(*args, **kwargs)

    def _get_result_class(self):
        return LayoutAnalysisResult

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: bool = False,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: Optional[bool] = True,
        skip_order_labels: Optional[List[str]] = None,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (Optional[float, dict], optional): The threshold for filtering out low-confidence predictions.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to None.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.
            layout_shape_mode (Optional[str], optional): The mode for layout shape. Defaults to "auto", [ "rect", "quad","poly", "auto"]. are supported.
            filter_overlap_boxes (Optional[bool], optional): Whether to filter out overlap boxes. Defaults to True.
            skip_order_labels (Optional[List[str]], optional): The labels to skip order. Defaults to None.

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data.instances
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1](datas)

        # do infer
        batch_preds = self.infer(batch_inputs)

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)
        # postprocess
        boxes = self.post_op(
            preds_list,
            datas,
            threshold=threshold if threshold is not None else self.threshold,
            layout_nms=layout_nms or self.layout_nms,
            layout_unclip_ratio=layout_unclip_ratio or self.layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
            or self.layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            filter_overlap_boxes=filter_overlap_boxes,
            skip_order_labels=skip_order_labels,
        )

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }

    @DetPredictor.register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
        self.target_size = target_size
        if isinstance(interp, int):
            interp = {
                0: "NEAREST",
                1: "LINEAR",
                2: "BICUBIC",
                3: "AREA",
                4: "LANCZOS4",
            }[interp]
        op = Resize(target_size=target_size[::-1], keep_ratio=keep_ratio, interp=interp)
        return op

    def build_to_batch(self):
        models_required_imgsize = [
            "PP-DocLayoutV2",
            "PP-DocLayoutV3",
        ]
        if any(name in self.model_name for name in models_required_imgsize):
            ordered_required_keys = (
                "img_size",
                "img",
                "scale_factors",
            )
        else:
            ordered_required_keys = ("img", "scale_factors")

        return ToBatch(ordered_required_keys=ordered_required_keys)

    def build_postprocess(self):
        if self.threshold is None:
            self.threshold = self.config.get("draw_threshold", 0.5)
        if not self.layout_nms:
            self.layout_nms = self.config.get("layout_nms", None)
        if self.layout_unclip_ratio is None:
            self.layout_unclip_ratio = self.config.get("layout_unclip_ratio", None)
        if self.layout_merge_bboxes_mode is None:
            self.layout_merge_bboxes_mode = self.config.get(
                "layout_merge_bboxes_mode", None
            )
        scale_size = getattr(self, "target_size", [800, 800])
        return LayoutAnalysisProcess(
            labels=self.config["label_list"], scale_size=scale_size
        )
