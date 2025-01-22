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

from ..utils import logging
from ..utils.flags import USE_NEW_INFERENCE, NEW_PREDICTOR

if USE_NEW_INFERENCE:
    logging.warning("=" * 20 + " Using pipelines_new " + "=" * 20)
    from .pipelines_new import create_pipeline, load_pipeline_config
else:
    from .pipelines import create_pipeline, load_pipeline_config
if NEW_PREDICTOR:
    logging.warning("=" * 20 + " Using models_new " + "=" * 20)
    from .models_new import create_predictor
else:
    from .models import create_predictor
from .utils.pp_option import PaddlePredictorOption
