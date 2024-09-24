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

from copy import deepcopy
from ..inference.predictors import create_predictor
from ..inference.utils.pp_option import PaddlePredictorOption
from ..utils.config import AttrDict


class Predictor(object):
    def __init__(self, config):
        model_name = config.Global.model
        predict_config = deepcopy(config.Predict)

        model_dir = predict_config.pop("model_dir", None)
        # if model_dir is None, using official
        model = model_name if model_dir is None else model_dir
        self.input_path = predict_config.pop("input_path")
        pp_option = PaddlePredictorOption(**predict_config.pop("kernel_option", {}))
        self.predictor = create_predictor(model, pp_option=pp_option, **predict_config)

    def predict(self):
        for res in self.predictor(self.input_path):
            res.print()


def build_predictor(config: AttrDict):
    """build predictor by config for dev"""
    return Predictor(config)
