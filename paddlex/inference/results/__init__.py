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

from .base import BaseResult
from .clas import TopkResult, MLClassResult
from .text_det import TextDetResult
from .text_rec import TextRecResult
from .table_rec import TableRecResult, StructureTableResult, TableResult
from .seal_rec import SealOCRResult
from .ocr import OCRResult
from .det import DetResult
from .seg import SegResult
from .formula_rec import FormulaRecResult
from .instance_seg import InstanceSegResult
from .ts import TSFcResult, TSAdResult, TSClsResult
from .warp import DocTrResult
from .chat_ocr import *
from .face_rec import FaceRecResult
