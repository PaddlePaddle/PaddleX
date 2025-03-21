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

import os
from typing import List, Tuple, Union

import numpy as np

from ...common.tokenizer.clip_tokenizer import CLIPTokenizer
from .common import LetterResize


class YOLOWorldProcessor(object):
    """Image and Text Processors for YOLO-World"""

    def __init__(
        self,
        model_dir,
        image_target_size: Union[Tuple[int], int] = (640, 640),
        image_mean: Union[float, List[float]] = [0.0, 0.0, 0.0],
        image_std: Union[float, List[float]] = [1.0, 1.0, 1.0],
        **kwargs,
    ):

        if isinstance(image_target_size, int):
            image_target_size = (image_target_size, image_target_size)
        if isinstance(image_mean, float):
            image_mean = [image_mean, image_mean, image_mean]
        if isinstance(image_std, float):
            image_std = [image_std, image_std, image_std]

        self.image_target_size = image_target_size
        self.image_mean = image_mean
        self.image_std = image_std

        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        assert os.path.isdir(tokenizer_dir), f"{tokenizer_dir} not exists."
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)

        self.resize_op = LetterResize(self.image_target_size, allow_scale_up=True)

        if isinstance(image_mean, (tuple, list)):
            self.image_mean = np.array(image_mean)
        if self.image_mean.ndim < 4:
            self.image_mean = self.image_mean.reshape(1, -1, 1, 1)

        if isinstance(image_std, (tuple, list)):
            self.image_std = np.array(image_std)
        if self.image_std.ndim < 4:
            self.image_std = self.image_std.reshape(1, -1, 1, 1)

    def __call__(
        self,
        images: List[np.ndarray],
        text: str,
        **kwargs,
    ):
        preprocess_results = self.process_image(images)
        preprocess_results.update(self.process_text(text))
        static_input_orders = [
            "attention_mask",
            "image",
            "input_ids",
            "pad_param",
            "scale_factor",
        ]
        result = [preprocess_results[k] for k in static_input_orders]

        return result

    def process_image(self, images):
        """Image preprocess for YOLO-World"""
        rescaled_images = self.resize_op(images)

        images = np.stack(
            [rescaled_image["image"] for rescaled_image in rescaled_images], axis=0
        )
        scale_factors = np.stack(
            [rescaled_image["scale_factor"] for rescaled_image in rescaled_images],
            axis=0,
        )
        pad_params = np.stack(
            [rescaled_image["pad_param"] for rescaled_image in rescaled_images], axis=0
        )

        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32) / 255.0
        images -= self.image_mean
        images /= self.image_std

        image_results = {
            "image": images,
            "scale_factor": scale_factors,
            "pad_param": pad_params[:, [3, 0]],
        }

        return image_results

    def process_text(self, text):

        text = text.strip().lower()
        words = [word.strip() for word in text.split(",")]
        words += [" "]
        tokenized_text = self.tokenizer(text=words, return_tensors="pd", padding=True)

        text_results = {
            "input_ids": tokenized_text["input_ids"].numpy(),
            "attention_mask": tokenized_text["attention_mask"].numpy(),
        }

        return text_results


class YOLOWorldPostProcessor(object):
    """PostProcessors for YOLO-World"""

    def __init__(
        self,
        threshold: float = 0.05,
        **kwargs,
    ):
        """Init Function for YOLO-World PostProcessor

        Args:
            threshold (float): threshold for low confidence bbox filtering.
        """
        self.threshold = threshold

    def __call__(
        self,
        pred_boxes,
        pred_nums,
        prompt,
        src_images,
        threshold=None,
        **kwargs,
    ):

        threshold = self.threshold if threshold is None else threshold

        split_index = np.cumsum(pred_nums)[:-1]
        pred_boxes = np.split(pred_boxes, split_index, axis=0)
        assert len(pred_boxes) == len(src_images)

        classnames = self.prompt_to_classnames(prompt)

        rst_boxes = []
        for pred_box, src_image in zip(pred_boxes, src_images):
            rst_boxes.append(
                self.postprocess(
                    pred_box,
                    classnames,
                    src_image,
                    threshold,
                )
            )

        return rst_boxes

    def postprocess(
        self,
        pred_boxes,
        classnames,
        src_image,
        threshold,
    ):
        """Post Process for prediction result of single image."""

        pred_boxes = pred_boxes[pred_boxes[:, 1] > threshold]
        H, W, *_ = src_image.shape

        pred_labels = pred_boxes[:, 0].astype(np.int32)
        pred_scores = pred_boxes[:, 1]
        pred_bboxes = pred_boxes[:, 2:]

        pred_bboxes[:, ::2] = np.clip(pred_bboxes[:, ::2], a_min=0, a_max=W)
        pred_bboxes[:, 1::2] = np.clip(pred_bboxes[:, 1::2], a_min=0, a_max=H)

        rst_bboxes = []
        for pred_label, pred_score, pred_bbox in zip(
            pred_labels, pred_scores, pred_bboxes
        ):
            rst_bboxes.append(
                {
                    "coordinate": pred_bbox.tolist(),
                    "label": classnames[pred_label],
                    "score": pred_score,
                }
            )

        return rst_bboxes

    def prompt_to_classnames(self, text):

        text = text.strip().lower()
        words = [word.strip() for word in text.split(",")]
        words += [" "]

        return words
