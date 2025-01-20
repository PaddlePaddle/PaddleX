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

from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

# [TODO] 待更新models_new到models
from ...models_new.keypoint_detection.result import KptResult

Number = Union[int, float]


class KeypointDetectionPipeline(BasePipeline):
    """Keypoint Detection pipeline"""

    entities = "human_keypoint_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        # create object detection model
        model_cfg = config["SubModules"]["ObjectDetection"]
        model_kwargs = {}
        if "threshold" in model_cfg:
            model_kwargs["threshold"] = model_cfg["threshold"]
        if "img_size" in model_cfg:
            model_kwargs["img_size"] = model_cfg["img_size"]
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
        self, input: str | list[str] | np.ndarray | list[np.ndarray], **kwargs
    ) -> KptResult:
        """Predicts image classification results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            KptResult: The predicted KeyPoint Detection results.
        """

        for det_res in self.det_model(input):
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
                    }
                )
            yield KptResult(single_img_res)
