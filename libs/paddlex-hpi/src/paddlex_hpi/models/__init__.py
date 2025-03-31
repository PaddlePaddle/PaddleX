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

from paddlex_hpi.models.anomaly_detection import UadPredictor
from paddlex_hpi.models.base import HPPredictor
from paddlex_hpi.models.face_recognition import FaceRecPredictor
from paddlex_hpi.models.formula_recognition import LaTeXOCRPredictor
from paddlex_hpi.models.general_recognition import ShiTuRecPredictor
from paddlex_hpi.models.image_classification import ClasPredictor
from paddlex_hpi.models.image_unwarping import WarpPredictor
from paddlex_hpi.models.instance_segmentation import InstanceSegPredictor
from paddlex_hpi.models.multilabel_classification import MLClasPredictor
from paddlex_hpi.models.object_detection import DetPredictor
from paddlex_hpi.models.semantic_segmentation import SegPredictor
from paddlex_hpi.models.table_recognition import TablePredictor
from paddlex_hpi.models.text_detection import TextDetPredictor
from paddlex_hpi.models.text_recognition import TextRecPredictor
from paddlex_hpi.models.ts_ad import TSAdPredictor
from paddlex_hpi.models.ts_cls import TSClsPredictor
from paddlex_hpi.models.ts_fc import TSFcPredictor

__all__ = [
    "UadPredictor",
    "HPPredictor",
    "FaceRecPredictor",
    "LaTeXOCRPredictor",
    "ShiTuRecPredictor",
    "ClasPredictor",
    "WarpPredictor",
    "InstanceSegPredictor",
    "MLClasPredictor",
    "DetPredictor",
    "SegPredictor",
    "TablePredictor",
    "TextDetPredictor",
    "TextRecPredictor",
    "TSAdPredictor",
    "TSClsPredictor",
    "TSFcPredictor",
]
