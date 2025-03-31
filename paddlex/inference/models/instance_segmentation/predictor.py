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

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ....modules.instance_segmentation.model_list import MODELS
from ....utils import logging
from ..object_detection import DetPredictor
from ..object_detection.processors import ReadImage, ToBatch
from .processors import InstanceSegPostProcess
from .result import InstanceSegResult


class InstanceSegPredictor(DetPredictor):
    """InstanceSegPredictor that inherits from DetPredictor."""

    entities = MODELS

    def __init__(self, *args, threshold: Optional[float] = None, **kwargs):
        """Initializes InstanceSegPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            threshold (Optional[float], optional): The threshold for filtering out low-confidence predictions.
                Defaults to None, in which case will use default from the config file.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)

        self.model_names_only_supports_batchsize_of_one = {
            "SOLOv2",
            "PP-YOLOE_seg-S",
            "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
            "Cascade-MaskRCNN-ResNet50-FPN",
        }
        if self.model_name in self.model_names_only_supports_batchsize_of_one:
            logging.warning(
                f"Instance Segmentation Models: \"{', '.join(list(self.model_names_only_supports_batchsize_of_one))}\" only supports prediction with a batch_size of one, "
                "if you set the predictor with a batch_size larger than one, no error will occur, however, it will actually inference with a batch_size of one, "
                f"which will lead to a slower inference speed. You are now using {self.config['Global']['model_name']}."
            )

        self.threshold = threshold

    def _get_result_class(self) -> type:
        """Returns the result class, InstanceSegResult.

        Returns:
            type: The InstanceSegResult class.
        """
        return InstanceSegResult

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

        # build infer
        infer = self.create_static_infer()

        # build postprocess op
        post_op = self.build_postprocess()

        return pre_ops, infer, post_op

    def build_to_batch(self):

        ordered_required_keys = (
            "img_size",
            "img",
            "scale_factors",
        )

        return ToBatch(ordered_required_keys=ordered_required_keys)

    def process(self, batch_data: List[Any], threshold: Optional[float] = None):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, box and mask
                for every instance of the batch. Keys include 'input_path', 'input_img', 'boxes' and 'masks'.
        """
        datas = batch_data.instances
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1](datas)

        # do infer
        if self.model_name in self.model_names_only_supports_batchsize_of_one:
            batch_preds = []
            for i in range(batch_inputs[0].shape[0]):
                batch_inputs_ = [
                    batch_input_[i][None, ...] for batch_input_ in batch_inputs
                ]
                batch_pred_ = self.infer(batch_inputs_)
                batch_preds.append(batch_pred_)
        else:
            batch_preds = self.infer(batch_inputs)

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)

        # postprocess
        boxes_masks = self.post_op(
            preds_list, datas, threshold if threshold is not None else self.threshold
        )

        return {
            "input_path": batch_data.input_paths,
            "page_index": batch_data.page_indexes,
            "input_img": [data["ori_img"] for data in datas],
            "boxes": [result["boxes"] for result in boxes_masks],
            "masks": [result["masks"] for result in boxes_masks],
        }

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The input predictions, which can be either a list of 3 or 4 elements.
                - When len(pred) == 4, it is expected to be in the format [boxes, class_ids, scores, masks],
                  compatible with SOLOv2 output.
                - When len(pred) == 3, it is expected to be in the format [boxes, box_nums, masks],
                  compatible with Instance Segmentation output.

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        box_idx_start = 0
        pred_box = []

        if isinstance(pred[0], list) and len(pred[0]) == 4:
            # Adapt to SOLOv2, which only support prediction with a batch_size of 1.
            pred_class_id = [[pred_[1], pred_[2]] for pred_ in pred]
            pred_mask = [pred_[3] for pred_ in pred]
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]
        if isinstance(pred[0], list) and len(pred[0]) == 3:
            # Adapt to PP-YOLOE_seg-S, which only support prediction with a batch_size of 1.
            return [
                {"boxes": np.array(pred[i][0]), "masks": np.array(pred[i][2])}
                for i in range(len(pred))
            ]

        pred_mask = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = pred[0][box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            np_masks = pred[2][box_idx_start:box_idx_end]
            pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        return [
            {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
            for i in range(len(pred_box))
        ]

    def build_postprocess(self):
        return InstanceSegPostProcess(
            threshold=self.config["draw_threshold"], labels=self.config["label_list"]
        )
