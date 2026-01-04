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

from typing import Any, List, Optional, Sequence

import numpy as np

from ....modules.keypoint_detection.model_list import MODELS
from ....utils import logging
from ...common.batch_sampler import ImageBatchSampler
from ..common import ToBatch
from ..object_detection import DetPredictor
from .processors import KptPostProcess, TopDownAffine
from .result import KptResult


class KptBatchSampler(ImageBatchSampler):
    # don't support to pass pdf file as input
    PDF_SUFFIX = []

    def sample(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        batch = []
        for input in inputs:
            if isinstance(input, (np.ndarray, dict)):
                batch.append(input)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            elif isinstance(input, str):
                file_path = (
                    self._download_from_url(input)
                    if input.startswith("http")
                    else input
                )
                file_list = self._get_files_list(file_path)
                for file_path in file_list:
                    batch.append(file_path)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            else:
                logging.warning(
                    f"Not supported input data type! Only `numpy.ndarray` and `str` are supported! So has been ignored: {input}."
                )
        if len(batch) > 0:
            yield batch


class KptPredictor(DetPredictor):

    entities = MODELS

    flip_perm = [  # The left-right joints exchange order list
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
    ]

    def __init__(
        self,
        *args,
        flip: bool = False,
        use_udp: Optional[bool] = None,
        **kwargs,
    ):
        """Keypoint Predictor

        Args:
            flip (bool): Whether to do flipping test. Default value is ``False``.
            use_udp (Optional[bool]): Whether to use unbiased data processing. Default value is ``None``.

        """
        self.flip = flip
        self.use_udp = use_udp
        super().__init__(*args, **kwargs)
        for op in self.pre_ops:
            if isinstance(op, TopDownAffine):
                self.input_size = op.input_size
                break
        if any([name in self.model_name for name in ["PP-TinyPose"]]):
            self.shift_heatmap = True
        else:
            self.shift_heatmap = False

    def _build_batch_sampler(self):
        return KptBatchSampler()

    def _get_result_class(self):
        return KptResult

    def _format_output(self, pred: Sequence[Any]) -> List[dict]:
        """Transform batch outputs into a list of single image output."""

        return [
            {
                "heatmap": res[0],
                "masks": res[1],
            }
            for res in zip(*pred)
        ]

    def flip_back(self, output_flipped, matched_parts):
        assert (
            output_flipped.ndim == 4
        ), "output_flipped should be [batch_size, num_joints, height, width]"

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped

    def process(self, batch_data: List[dict]):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names
                for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        datas = batch_data
        # preprocess
        for pre_op in self.pre_ops[:-1]:
            datas = pre_op(datas)

        # use `ToBatch` format batch inputs
        batch_inputs = self.pre_ops[-1]([data["img"] for data in datas])

        # do infer
        batch_preds = self.infer(batch_inputs)

        if self.flip:
            # flip w
            batch_inputs[0] = np.flip(batch_inputs[0], axis=3)
            preds_flipped = self.infer(batch_inputs)

            output_flipped = self.flip_back(preds_flipped[0], self.flip_perm)
            if self.shift_heatmap:
                output_flipped[:, :, :, 1:] = output_flipped.copy()[:, :, :, 0:-1]
            batch_preds[0] = (batch_preds[0] + output_flipped) * 0.5

        # process a batch of predictions into a list of single image result
        preds_list = self._format_output(batch_preds)

        # postprocess
        keypoints = self.post_op(preds_list, datas)

        return {
            "input_path": [data.get("img_path", None) for data in datas],
            "input_img": [data["ori_img"] for data in datas],
            "kpts": keypoints,
        }

    @DetPredictor.register("TopDownEvalAffine")
    def build_topdown_affine(self, trainsize, use_udp=False):
        return TopDownAffine(
            input_size=trainsize,
            use_udp=use_udp if self.use_udp is None else self.use_udp,
        )

    def build_to_batch(self):
        return ToBatch()

    def build_postprocess(self):
        return KptPostProcess(use_dark=True)
