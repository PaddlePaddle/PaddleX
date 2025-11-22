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

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ....modules.object_detection.model_list import MODELS
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ImageBatchSampler
from ..base import BasePredictor
from .processors import (
    DetPad,
    DetPostProcess,
    Normalize,
    PadStride,
    ReadImage,
    Resize,
    ToBatch,
    ToCHWImage,
    WarpAffine,
)
from .result import DetResult
from .utils import STATIC_SHAPE_MODEL_LIST


class DetPredictor(BasePredictor):

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        *args,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
        **kwargs,
    ):
        """Initializes DetPredictor.
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
        super().__init__(*args, **kwargs)

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

        if layout_unclip_ratio is not None:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or Dict, but got {type(layout_unclip_ratio)}."
                )

        if layout_merge_bboxes_mode is not None:
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'] or a dict, but got {layout_merge_bboxes_mode}"

        self.img_size = img_size
        self.threshold = threshold
        self.layout_nms = layout_nms
        self.layout_unclip_ratio = layout_unclip_ratio
        self.layout_merge_bboxes_mode = layout_merge_bboxes_mode
        self.pre_ops, self.infer, self.post_op = self._build()

    def _build_batch_sampler(self):
        return ImageBatchSampler()

    def _get_result_class(self):
        return DetResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        # build preprocess ops
        pre_ops = [ReadImage(format="RGB")]
        for cfg in self.config["Preprocess"]:
            tf_key = cfg["type"]
            func = self._FUNC_MAP[tf_key]
            cfg.pop("type")
            args = cfg
            op = func(self, **args) if args else func(self)
            if op:
                pre_ops.append(op)
        pre_ops.append(self.build_to_batch())
        if self.img_size is not None:
            if isinstance(pre_ops[1], Resize):
                pre_ops.pop(1)
            pre_ops.insert(1, self.build_resize(self.img_size, False, 2))

        # build infer
        if self._use_static_model:
            infer = self.create_static_infer()
        else:
            if self.model_name not in []:
                raise RuntimeError(
                    f"There is no dynamic graph implementation for model {repr(self.model_name)}."
                )

        # build postprocess op
        post_op = self.build_postprocess()

        return pre_ops, infer, post_op

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The input predictions, which can be either a list of 3 or 4 elements.
                - When len(pred) == 4, it is expected to be in the format [boxes, class_ids, scores, masks],
                  compatible with SOLOv2 output.
                - When len(pred) == 3, it is expected to be in the format [boxes, box_nums, masks],
                  compatible with Instance Segmentation output.
                - When len(pred) >= 2 and pred[2] exists as batch_inds, use batch_inds for grouping.

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        box_idx_start = 0
        pred_box = []

        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]

        # Fix for multi-table cell detection issue:
        # When multiple table crops are processed in a single batch, post-NMS ordering
        # of detections can shift, causing cell assignments to no longer align with
        # their original tables. Solution: group by batch_idx instead of relying on
        # the order of flattened predictions.
        # Reference: https://github.com/PaddlePaddle/PaddleX/issues/17133
        
        # Check if batch_inds is available (for RT-DETR and similar models)
        # RT-DETR models may output batch_inds as a separate output from raw inference
        batch_inds = None
        
        # First, try to find batch_inds in the raw RT-DETR outputs
        # For RT-DETR, batch_inds might be in a separate output
        # We need to distinguish it from masks (which are 3D arrays for Instance Segmentation)
        # Check all outputs after box_nums for potential batch_inds
        for i in range(2, len(pred)):
            candidate = pred[i]
            # batch_inds should be 1D array of integers matching boxes length
            # masks for Instance Segmentation are typically 3D or have different shape
            if (
                isinstance(candidate, np.ndarray)
                and candidate.ndim == 1
                and candidate.dtype in (np.int32, np.int64, np.int8, np.int16)
                and len(candidate) == len(pred[0])
                and len(pred[1]) > 0
                and candidate.max() < len(pred[1])  # batch_id should be < batch_size
                and candidate.min() >= 0  # batch_id should be >= 0
            ):
                batch_inds = candidate.astype(np.int32)
                break  # Found batch_inds, stop searching
        
        # If batch_inds is not found in outputs, construct it from box_nums
        # This assumes boxes are ordered correctly before any NMS or reordering
        if batch_inds is None and len(pred) >= 2 and len(pred[1]) > 0:
            box_nums = pred[1]
            if isinstance(box_nums, np.ndarray) and box_nums.ndim == 1:
                total_boxes = len(pred[0])
                expected_total = int(box_nums.sum())
                # Only construct batch_inds if the total matches (boxes haven't been reordered)
                if total_boxes == expected_total:
                    batch_inds = np.zeros(total_boxes, dtype=np.int32)
                    box_idx = 0
                    for batch_idx, num_boxes in enumerate(box_nums):
                        num_boxes = int(num_boxes)
                        if box_idx + num_boxes <= total_boxes:
                            batch_inds[box_idx : box_idx + num_boxes] = batch_idx
                            box_idx += num_boxes
                        else:
                            # If mismatch, don't use batch_inds
                            batch_inds = None
                            break

        # Use batch_inds for grouping if available
        # This ensures correct grouping even when post-NMS reorders detections
        if batch_inds is not None:
            unique_batch_ids = np.unique(batch_inds)
            pred_box = []
            # Find mask output if exists (for Instance Segmentation)
            # Masks are typically not 1D integer arrays (unlike batch_inds)
            mask_output_idx = None
            if len(pred) >= 3:
                for i in range(2, len(pred)):
                    candidate = pred[i]
                    # If this is not the batch_inds we found, and it looks like masks
                    if (
                        isinstance(candidate, np.ndarray)
                        and not (candidate.ndim == 1 and candidate.dtype in (np.int32, np.int64, np.int8, np.int16) and len(candidate) == len(pred[0]) and candidate.max() < len(pred[1]) and candidate.min() >= 0)
                        and len(candidate) == len(pred[0])
                    ):
                        mask_output_idx = i
                        break
            
            pred_mask = [] if mask_output_idx is not None else None
            
            # Group by batch_idx instead of slicing (fixes post-NMS ordering issue)
            for batch_id in unique_batch_ids:
                mask = batch_inds == batch_id
                np_boxes = pred[0][mask]
                pred_box.append(np_boxes)
                if pred_mask is not None and mask_output_idx is not None:
                    np_masks = pred[mask_output_idx][mask]
                    pred_mask.append(np_masks)
        else:
            # Fallback to original box_nums slicing method
            if len(pred) == 3:
                # Adapt to Instance Segmentation
                pred_mask = []
            for idx in range(len(pred[1])):
                np_boxes_num = pred[1][idx]
                box_idx_end = box_idx_start + np_boxes_num
                np_boxes = pred[0][box_idx_start:box_idx_end]
                pred_box.append(np_boxes)
                if len(pred) == 3:
                    np_masks = pred[2][box_idx_start:box_idx_end]
                    pred_mask.append(np_masks)
                box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]

    def process(
        self,
        batch_data: List[Any],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: bool = False,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[Union[str, dict]] = None,
    ):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            threshold (Optional[float, dict], optional): The threshold for filtering out low-confidence predictions.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to None.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
            layout_merge_bboxes_mode (Optional[Union[str, dict]], optional): The mode for merging bounding boxes. Defaults to None.

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
        )

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": boxes,
        }

    @register("Resize")
    def build_resize(self, target_size, keep_ratio=False, interp=2):
        assert target_size
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

    @register("NormalizeImage")
    def build_normalize(
        self,
        norm_type=None,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        is_scale=True,
    ):
        if is_scale:
            scale = 1.0 / 255.0
        else:
            scale = 1
        if not norm_type or norm_type == "none":
            norm_type = "mean_std"
        if norm_type != "mean_std":
            mean = 0
            std = 1
        return Normalize(scale=scale, mean=mean, std=std)

    @register("Permute")
    def build_to_chw(self):
        return ToCHWImage()

    @register("Pad")
    def build_pad(self, fill_value=None, size=None):
        if fill_value is None:
            fill_value = [127.5, 127.5, 127.5]
        if size is None:
            size = [3, 640, 640]
        return DetPad(size=size, fill_value=fill_value)

    @register("PadStride")
    def build_pad_stride(self, stride=32):
        return PadStride(stride=stride)

    @register("WarpAffine")
    def build_warp_affine(self, input_h=512, input_w=512, keep_res=True):
        return WarpAffine(input_h=input_h, input_w=input_w, keep_res=keep_res)

    def build_to_batch(self):
        models_required_imgsize = [
            "DETR",
            "DINO",
            "RCNN",
            "YOLOv3",
            "CenterNet",
            "BlazeFace",
            "BlazeFace-FPN-SSH",
            "PP-DocLayout-L",
            "PP-DocLayout_plus-L",
            "PP-DocBlockLayout",
            "PP-DocLayoutV2",
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
        return DetPostProcess(labels=self.config["label_list"])
