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
from importlib import import_module

from .anomaly_detection import UadDatasetChecker, UadEvaluator, UadExportor, UadTrainer
from .base import build_dataset_checker, build_evaluater, build_exportor, build_trainer
from .face_recognition import (
    FaceRecDatasetChecker,
    FaceRecEvaluator,
    FaceRecExportor,
    FaceRecTrainer,
)
from .general_recognition import (
    ShiTuRecDatasetChecker,
    ShiTuRecEvaluator,
    ShiTuRecExportor,
    ShiTuRecTrainer,
)
from .image_classification import (
    ClsDatasetChecker,
    ClsEvaluator,
    ClsExportor,
    ClsTrainer,
)
from .instance_segmentation import (
    COCOInstSegDatasetChecker,
    InstanceSegEvaluator,
    InstanceSegExportor,
    InstanceSegTrainer,
)
from .multilabel_classification import (
    MLClsDatasetChecker,
    MLClsEvaluator,
    MLClsExportor,
    MLClsTrainer,
)
from .object_detection import COCODatasetChecker, DetEvaluator, DetExportor, DetTrainer
from .semantic_segmentation import (
    SegDatasetChecker,
    SegEvaluator,
    SegExportor,
    SegTrainer,
)
from .table_recognition import (
    TableRecDatasetChecker,
    TableRecEvaluator,
    TableRecExportor,
    TableRecTrainer,
)
from .text_detection import (
    TextDetDatasetChecker,
    TextDetEvaluator,
    TextDetExportor,
    TextDetTrainer,
)
from .text_recognition import (
    TextRecDatasetChecker,
    TextRecEvaluator,
    TextRecExportor,
    TextRecTrainer,
)
from .ts_anomaly_detection import (
    TSADDatasetChecker,
    TSADEvaluator,
    TSADExportor,
    TSADTrainer,
)
from .ts_classification import (
    TSCLSDatasetChecker,
    TSCLSEvaluator,
    TSCLSExportor,
    TSCLSTrainer,
)
from .ts_forecast import TSFCDatasetChecker, TSFCEvaluator, TSFCTrainer

module_3d_bev_detection = import_module(".3d_bev_detection", "paddlex.modules")
BEVFusionDatasetChecker = getattr(module_3d_bev_detection, "BEVFusionDatasetChecker")
BEVFusionTrainer = getattr(module_3d_bev_detection, "BEVFusionTrainer")
BEVFusionEvaluator = getattr(module_3d_bev_detection, "BEVFusionEvaluator")
BEVFusionExportor = getattr(module_3d_bev_detection, "BEVFusionExportor")

from .keypoint_detection import (
    KeypointDatasetChecker,
    KeypointEvaluator,
    KeypointExportor,
    KeypointTrainer,
)
from .multilingual_speech_recognition import (
    WhisperDatasetChecker,
    WhisperEvaluator,
    WhisperExportor,
    WhisperTrainer,
)
from .video_classification import (
    VideoClsDatasetChecker,
    VideoClsEvaluator,
    VideoClsExportor,
    VideoClsTrainer,
)
from .video_detection import (
    VideoDetDatasetChecker,
    VideoDetEvaluator,
    VideoDetExportor,
    VideoDetTrainer,
)
