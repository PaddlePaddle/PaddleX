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

from typing import Any, Union, Dict, List, Tuple
import numpy as np

from ....utils.func_register import FuncRegister
from ....modules.table_recognition.model_list import MODELS
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import (
    Resize,
    ResizeByLong,
    Normalize,
    ToCHWImage,
    ToBatch,
    StaticInfer,
)
from ..base import BasicPredictor
from .processors import Pad, TableLabelDecode
from .result import TableRecResult


class TablePredictor(BasicPredictor):
    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(*args, **kwargs)
        self.preprocessors, self.infer, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        return TableRecResult

    def _build(self) -> Tuple:
        preprocessors = []
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            op = func(self, **args) if args else func(self)
            if op:
                preprocessors.append(op)
        preprocessors.append(ToBatch())

        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        postprocessors = TableLabelDecode(
            model_name=self.config["Global"]["model_name"],
            merge_no_span_structure=self.config["PreProcess"]["transform_ops"][1][
                "TableLabelEncode"
            ]["merge_no_span_structure"],
            dict_character=self.config["PostProcess"]["character_dict"],
        )
        return preprocessors, infer, postprocessors

    def process(self, batch_data: List[Union[str, np.ndarray]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """
        batch_raw_imgs = self.preprocessors[0](imgs=batch_data)  # ReadImage
        ori_shapes = []
        for s in range(len(batch_raw_imgs)):
            ori_shapes.append([batch_raw_imgs[s].shape[1], batch_raw_imgs[s].shape[0]])
        batch_imgs = self.preprocessors[1](imgs=batch_raw_imgs)  # ResizeByLong
        batch_imgs = self.preprocessors[2](imgs=batch_imgs)  # Normalize
        pad_results = self.preprocessors[3](imgs=batch_imgs)  # Pad
        pad_imgs = []
        padding_sizes = []
        for pad_img, padding_size in pad_results:
            pad_imgs.append(pad_img)
            padding_sizes.append(padding_size)
        batch_imgs = self.preprocessors[4](imgs=pad_imgs)  # ToCHWImage
        x = self.preprocessors[5](imgs=batch_imgs)  # ToBatch

        batch_preds = self.infer(x=x)

        table_result = self.postprocessors(
            pred=batch_preds,
            img_size=padding_sizes,
            ori_img_size=ori_shapes,
        )

        table_result_bbox = []
        table_result_structure = []
        table_result_structure_score = []
        for i in range(len(table_result)):
            table_result_bbox.append(table_result[i]["bbox"])
            table_result_structure.append(table_result[i]["structure"])
            table_result_structure_score.append(table_result[i]["structure_score"])

        final_result = {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "bbox": table_result_bbox,
            "structure": table_result_structure,
            "structure_score": table_result_structure_score,
        }

        return final_result

    @register("DecodeImage")
    def build_readimg(self, channel_first=False, img_mode="BGR"):
        assert channel_first is False
        assert img_mode == "BGR"
        return ReadImage(format=img_mode)

    @register("TableLabelEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("TableBoxEncode")
    def foo(self, *args, **kwargs):
        return None

    @register("ResizeTableImage")
    def build_resize_table(self, max_len=488, resize_bboxes=True):
        return ResizeByLong(target_long_edge=max_len)

    @register("NormalizeImage")
    def build_normalize(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        scale=1 / 255,
        order="hwc",
    ):
        return Normalize(mean=mean, std=std)

    @register("PaddingTableImage")
    def build_padding(self, size=[488, 448], pad_value=0):
        return Pad(target_size=size[0], val=pad_value)

    @register("ToCHWImage")
    def build_to_chw(self):
        return ToCHWImage()

    @register("KeepKeys")
    def foo(self, *args, **kwargs):
        return None

    def _pack_res(self, single):
        keys = ["input_path", "bbox", "structure"]
        return TableRecResult({key: single[key] for key in keys})
