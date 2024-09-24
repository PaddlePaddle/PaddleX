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


from .base import (
    build_dataset_checker,
    build_trainer,
    build_evaluater,
    build_exportor,
)

from .predictor import build_predictor

from .image_classification import (
    ClsDatasetChecker,
    ClsTrainer,
    ClsEvaluator,
    ClsExportor,
)

from .multilabel_classification import (
    MLClsDatasetChecker,
    MLClsTrainer,
    MLClsEvaluator,
    MLClsExportor,
)

from .anomaly_detection import (
    UadDatasetChecker,
    UadTrainer,
    UadEvaluator,
    UadExportor,
)
from .general_recognition import (
    ShiTuRecDatasetChecker,
    ShiTuRecTrainer,
    ShiTuRecEvaluator,
    ShiTuRecExportor,
)
from .object_detection import (
    COCODatasetChecker,
    DetTrainer,
    DetEvaluator,
    DetExportor,
)
from .text_detection import (
    TextDetDatasetChecker,
    TextDetTrainer,
    TextDetEvaluator,
    TextDetExportor,
)
from .text_recognition import (
    TextRecDatasetChecker,
    TextRecTrainer,
    TextRecEvaluator,
    TextRecExportor,
)
from .table_recognition import (
    TableRecDatasetChecker,
    TableRecTrainer,
    TableRecEvaluator,
    TableRecExportor,
)
from .semantic_segmentation import (
    SegDatasetChecker,
    SegTrainer,
    SegEvaluator,
    SegExportor,
)
from .instance_segmentation import (
    COCOInstSegDatasetChecker,
    InstanceSegTrainer,
    InstanceSegEvaluator,
    InstanceSegExportor,
)
from .ts_anomaly_detection import (
    TSADDatasetChecker,
    TSADTrainer,
    TSADEvaluator,
    TSADExportor,
)
from .ts_classification import (
    TSCLSDatasetChecker,
    TSCLSTrainer,
    TSCLSEvaluator,
    TSCLSExportor,
)

from .ts_forecast import TSFCDatasetChecker, TSFCTrainer, TSFCEvaluator
