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

from typing import Any, Dict, Optional, Union, Tuple, List

import cv2
import numpy as np

from ...common.batch_sampler import VideoBatchSampler
from ...utils.pp_option import PaddlePredictorOption
from ...utils.io import VideoReader
from ..base import BasePipeline
from .trackers import JDETracker
from .mot import JDEMOT
from .result import MOTVideoResult


class MultiObjectTrackingPipeline(BasePipeline):

    entities = "multiobject_tracking"

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

        # create detector
        model_cfg = config["SubModules"]["Detector"]
        model_kwargs = {}
        if "threshold" in model_cfg:
            model_kwargs["threshold"] = model_cfg["threshold"]
            self.det_threshold = model_cfg["threshold"]
        if "imgsz" in model_cfg:
            model_kwargs["imgsz"] = model_cfg["imgsz"]
        self.detector = self.create_model(model_cfg, **model_kwargs)
        self.labels = self.detector.labels
        self.num_classes = len(self.labels)

        # create reid
        if config["SubModules"].get("ReID", None) is not None:
            model_cfg = config["SubModules"]["ReID"]
            self.reid = self.create_model(model_cfg)
        else:
            self.reid = None

        # create tracker
        if config["SubModules"].get("Tracker", None) is not None:
            model_cfg = config["SubModules"]["Tracker"]
            if model_cfg["module_name"] == "JDETracker":
                model_cfg.pop("module_name")
                self.tracker = JDETracker(**model_cfg)
        else:
            self.tracker = None

        # create mot
        if config["SubModules"].get("MOT", None) is not None:
            model_cfg = config["SubModules"]["MOT"]
            if model_cfg["module_name"] == "JDEMOT":
                model_cfg.pop("module_name")
                self.mot = JDEMOT(
                    detector=self.detector,
                    reid=self.reid,
                    tracker=self.tracker,
                    num_classes=self.num_classes,
                    **model_cfg
                )
        else:
            self.mot = None

        # initialize video reader and batch sampler
        self.video_reader = VideoReader(backend="opencv")
        self.batch_sampler = VideoBatchSampler(batch_size=1)

    def predict(self, input: Union[str, List[str]], **kwargs):
        for videos in self.batch_sampler(input):
            for video_path in videos:
                video_results = []
                for frame_id, frame in enumerate(self.video_reader.read(video_path)):
                    single_frame_results = []
                    online_results = self.mot.tracking(frame, frame_id=frame_id)
                    for cls_id in range(self.num_classes):
                        for track in online_results[cls_id]:
                            single_frame_results.append(
                                {
                                    "cls_id": cls_id,
                                    "label": self.labels[cls_id],
                                    "track_id": int(track.track_id),
                                    "score": float(track.score),
                                    "coordinate": list(track.tlbr),
                                }
                            )
                    video_results.append(
                        {"frame_id": frame_id, "track_res": single_frame_results}
                    )
                yield MOTVideoResult(
                    {"input_path": video_path, "result": video_results}
                )
