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


from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ...utils import errors
from ..utils.hpi import HPIConfig
from ..utils.official_models import official_models

# from .table_recognition import TablePredictor
# from .general_recognition import ShiTuRecPredictor
from .anomaly_detection import UadPredictor
from .base import BasePredictor
from .doc_vlm import DocVLMPredictor
from .face_feature import FaceFeaturePredictor
from .formula_recognition import FormulaRecPredictor
from .image_classification import ClasPredictor
from .image_feature import ImageFeaturePredictor
from .image_multilabel_classification import MLClasPredictor
from .image_unwarping import WarpPredictor
from .instance_segmentation import InstanceSegPredictor
from .keypoint_detection import KptPredictor
from .m_3d_bev_detection import BEVDet3DPredictor

# from .face_recognition import FaceRecPredictor
from .multilingual_speech_recognition import WhisperPredictor
from .object_detection import DetPredictor
from .open_vocabulary_detection import OVDetPredictor
from .open_vocabulary_segmentation import OVSegPredictor
from .semantic_segmentation import SegPredictor
from .table_structure_recognition import TablePredictor
from .text_detection import TextDetPredictor
from .text_recognition import TextRecPredictor
from .ts_anomaly_detection import TSAdPredictor
from .ts_classification import TSClsPredictor
from .ts_forecasting import TSFcPredictor
from .video_classification import VideoClasPredictor
from .video_detection import VideoDetPredictor


def create_predictor(
    model_name: str,
    model_dir: Optional[str] = None,
    device=None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    *args,
    **kwargs,
) -> BasePredictor:
    if model_dir is None:
        assert (
            model_name in official_models
        ), f"The model ({model_name}) is not supported! Please using directory of local model files or model name supported by PaddleX!"
        model_dir = official_models[model_name]
    else:
        assert Path(model_dir).exists(), f"{model_dir} is not exists!"
        model_dir = Path(model_dir)
    config = BasePredictor.load_config(model_dir)
    assert (
        model_name == config["Global"]["model_name"]
    ), f"Model name mismatch，please input the correct model dir."
    return BasePredictor.get(model_name)(
        model_dir=model_dir,
        config=config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_config=hpi_config,
        *args,
        **kwargs,
    )
