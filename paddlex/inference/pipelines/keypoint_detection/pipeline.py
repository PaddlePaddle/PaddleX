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

from ....utils.deps import pipeline_requires_extra
from ...models.keypoint_detection.result import KptResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline

Number = Union[int, float]


class _KeypointDetectionPipeline(BasePipeline):
    """Keypoint Detection pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        # create object detection model
        model_cfg = config["SubModules"]["ObjectDetection"]
        model_kwargs = {}
        self.det_threshold = None
        if "threshold" in model_cfg:
            model_kwargs["threshold"] = model_cfg["threshold"]
            self.det_threshold = model_cfg["threshold"]
        if "imgsz" in model_cfg:
            model_kwargs["imgsz"] = model_cfg["imgsz"]
        self.det_model = self.create_model(model_cfg, **model_kwargs)

        # create keypoint detection model
        model_cfg = config["SubModules"]["KeypointDetection"]
        model_kwargs = {}
        if "flip" in model_cfg:
            model_kwargs["flip"] = model_cfg["flip"]
        if "use_udp" in model_cfg:
            model_kwargs["use_udp"] = model_cfg["use_udp"]
        self.kpt_model = self.create_model(model_cfg, **model_kwargs)

        self.kpt_input_size = self.kpt_model.input_size

    def _box_xyxy2cs(
        self, bbox: Union[Number, np.ndarray], padding: float = 1.25
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert bounding box from (x1, y1, x2, y2) to center and scale.

        Args:
            bbox (Union[Number, np.ndarray]): The bounding box coordinates (x1, y1, x2, y2).
            padding (float): The padding factor to adjust the scale of the bounding box.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The center and scale of the bounding box.
        """
        x1, y1, x2, y2 = bbox[:4]
        center = np.array([x1 + x2, y1 + y2]) * 0.5

        # reshape bbox to fixed aspect ratio
        aspect_ratio = self.kpt_input_size[0] / self.kpt_input_size[1]
        w, h = x2 - x1, y2 - y1
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio

        scale = np.array([w, h]) * padding

        return center, scale

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        det_threshold: Optional[float] = None,
        **kwargs,
    ) -> KptResult:
        """Predicts image classification results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            det_threshold (float): The detection threshold. Defaults to None.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            KptResult: The predicted KeyPoint Detection results.
        """
        det_threshold = self.det_threshold if det_threshold is None else det_threshold
        for det_res in self.det_model(input, threshold=det_threshold):
            ori_img, img_path = det_res["input_img"], det_res["input_path"]
            single_img_res = {"input_path": img_path, "input_img": ori_img, "boxes": []}
            for box in det_res["boxes"]:
                center, scale = self._box_xyxy2cs(box["coordinate"])
                kpt_res = next(
                    self.kpt_model(
                        {
                            "img": ori_img,
                            "center": center,
                            "scale": scale,
                        }
                    )
                )
                single_img_res["boxes"].append(
                    {
                        "coordinate": box["coordinate"],
                        "det_score": box["score"],
                        "keypoints": kpt_res["kpts"][0]["keypoints"],
                        "kpt_score": kpt_res["kpts"][0]["kpt_score"],
                    }
                )
            yield KptResult(single_img_res)


@pipeline_requires_extra("cv")
class KeypointDetectionPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "human_keypoint_detection"

    @property
    def _pipeline_cls(self):
        return _KeypointDetectionPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["ObjectDetection"].get("batch_size", 1)
