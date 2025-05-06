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

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from .result import AttributeRecResult


class _AttributeRecPipeline(BasePipeline):
    """Attribute Rec Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ):
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        self.det_model = self.create_model(config["SubModules"]["Detection"])
        self.cls_model = self.create_model(config["SubModules"]["Classification"])
        self._crop_by_boxes = CropByBoxes()
        self._img_reader = ReadImage(format="BGR")

        self.det_threshold = config["SubModules"]["Detection"].get("threshold", 0.5)
        self.cls_threshold = config["SubModules"]["Classification"].get(
            "threshold", 0.7
        )

        self.batch_sampler = ImageBatchSampler(
            batch_size=config["SubModules"]["Detection"]["batch_size"]
        )
        self.img_reader = ReadImage(format="BGR")

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        det_threshold: float = None,
        cls_threshold: Union[float, dict, list, None] = None,
        **kwargs
    ):
        det_threshold = self.det_threshold if det_threshold is None else det_threshold
        cls_threshold = self.cls_threshold if cls_threshold is None else cls_threshold
        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            raw_imgs = self.img_reader(batch_data.instances)
            all_det_res = list(self.det_model(raw_imgs, threshold=det_threshold))
            for input_path, input_data, raw_img, det_res in zip(
                batch_data.input_paths, batch_data.instances, raw_imgs, all_det_res
            ):
                cls_res = self.get_cls_result(raw_img, det_res, cls_threshold)
                yield self.get_final_result(input_path, raw_img, det_res, cls_res)

    def get_cls_result(self, raw_img, det_res, cls_threshold):
        subs_of_img = list(self._crop_by_boxes(raw_img, det_res["boxes"]))
        img_list = [img["img"] for img in subs_of_img]
        all_cls_res = list(self.cls_model(img_list, threshold=cls_threshold))
        output = {"label": [], "score": []}
        for res in all_cls_res:
            output["label"].append(res["label_names"])
            output["score"].append(res["scores"])
        return output

    def get_final_result(self, input_path, raw_img, det_res, rec_res):
        single_img_res = {"input_path": input_path, "input_img": raw_img, "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            cls_scores = rec_res["score"][i]
            labels = rec_res["label"][i]
            single_img_res["boxes"].append(
                {
                    "labels": labels,
                    "cls_scores": cls_scores,
                    "det_score": obj["score"],
                    "coordinate": obj["coordinate"],
                }
            )
        return AttributeRecResult(single_img_res)


class AttributeRecPipeline(AutoParallelImageSimpleInferencePipeline):
    @property
    def _pipeline_cls(self):
        return _AttributeRecPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["Detection"]["batch_size"]


@pipeline_requires_extra("cv")
class PedestrianAttributeRecPipeline(AttributeRecPipeline):
    entities = "pedestrian_attribute_recognition"


@pipeline_requires_extra("cv")
class VehicleAttributeRecPipeline(AttributeRecPipeline):
    entities = "vehicle_attribute_recognition"
