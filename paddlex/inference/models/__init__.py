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


from pathlib import Path
from typing import Any, Dict, Optional

from ..utils.official_models import official_models
from .base import BasePredictor, BasicPredictor
from .image_classification import ClasPredictor
from .text_detection import TextDetPredictor
from .text_recognition import TextRecPredictor
from .table_recognition import TablePredictor
from .object_detection import DetPredictor
from .instance_segmentation import InstanceSegPredictor
from .semantic_segmentation import SegPredictor
from .general_recognition import ShiTuRecPredictor
from .ts_fc import TSFcPredictor
from .ts_ad import TSAdPredictor
from .ts_cls import TSClsPredictor
from .image_unwarping import WarpPredictor
from .multilabel_classification import MLClasPredictor
from .anomaly_detection import UadPredictor
from .formula_recognition import LaTeXOCRPredictor


def _create_hp_predictor(
    model_name, model_dir, device, config, hpi_params, *args, **kwargs
):
    try:
        from paddlex_hpi.models import HPPredictor
    except ModuleNotFoundError:
        raise RuntimeError(
            "The PaddleX HPI plugin is not properly installed, and the high-performance model inference features are not available."
        ) from None
    return HPPredictor.get(model_name)(
        model_dir=model_dir,
        config=config,
        device=device,
        *args,
        hpi_params=hpi_params,
        **kwargs,
    )


def create_predictor(
    model: str,
    device=None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> BasePredictor:
    model_dir = check_model(model)
    config = BasePredictor.load_config(model_dir)
    model_name = config["Global"]["model_name"]
    if use_hpip:
        return _create_hp_predictor(
            model_name=model_name,
            model_dir=model_dir,
            config=config,
            hpi_params=hpi_params,
            device=device,
            *args,
            **kwargs,
        )
    else:
        return BasicPredictor.get(model_name)(
            model_dir=model_dir,
            config=config,
            device=device,
            pp_option=pp_option,
            *args,
            **kwargs,
        )


def check_model(model):
    if Path(model).exists():
        return Path(model)
    elif model in official_models:
        return official_models[model]
    else:
        raise Exception(
            f"The model ({model}) is no exists! Please using directory of local model files or model name supported by PaddleX!"
        )
