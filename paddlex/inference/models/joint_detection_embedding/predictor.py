# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from typing import Any, List

from ....modules.joint_detection_embedding.model_list import MODELS
from ..object_detection import DetPredictor
from ..object_detection.processors import ToBatch

from .result import JDEResult


class JDEPredictor(DetPredictor):

    entities = MODELS

    def __init__(self, *args, **kwargs):
        """Initializes DetPredictor.
        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        if "batch_size" in kwargs:
            assert kwargs["batch_size"] == 1, "JDEPredictor only supports batch_size=1"
        super().__init__(*args, **kwargs)

    def _get_result_class(self):
        return JDEResult

    def process(self, batch_data: List[Any]):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
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
        pred_dets, pred_embs = self.infer(batch_inputs)

        return {
            "input_path": batch_data.input_paths,
            "input_img": [data["ori_img"] for data in datas],
            "pred_dets": [pred_dets],
            "pred_embs": [pred_embs],
        }

    def build_to_batch(self):
        models_required_imgsize = ["FairMOT"]
        if any(name in self.model_name for name in models_required_imgsize):
            ordered_required_keys = (
                "img_size",
                "img",
                "scale_factors",
            )
        else:
            ordered_required_keys = ("img", "scale_factors")

        return ToBatch(ordered_required_keys=ordered_required_keys)
