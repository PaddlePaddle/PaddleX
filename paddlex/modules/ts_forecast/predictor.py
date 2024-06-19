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

from typing import Union
from ...utils import logging
from ..base.build_model import build_model
from ..base.predictor import BasePredictor
from ...utils.errors import raise_unsupported_api_error, raise_model_not_found_error
from .support_models import SUPPORT_MODELS


class TSFCPredictor(BasePredictor):
    """ TS Forecast Model Predictor """
    support_models = SUPPORT_MODELS

    def __init__(self, config):
        """Initialize the instance.

        Args:
            config (AttrDict):  PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        self.global_config = config.Global
        self.predict_config = config.Predict

        config_path = self.get_config_path()
        self.pdx_config, self.pdx_model = build_model(
            self.global_config.model, config_path=config_path)

    def get_config_path(self) -> Union[str, None]:
        """
        get config path

        Returns:
            config_path (str): The path to the config

        """
        model_dir = self.predict_config.model_dir
        if Path(model_dir).exists():
            config_path = Path(model_dir).parent.parent / "config.yaml"
            if config_path.exists():
                return config_path
            else:
                logging.warning(
                    f"The config file(`{config_path}`) related to model weight file(`{self.predict_config.model_dir}`) \
is not exist, use default instead.")
        else:
            raise_model_not_found_error(model_dir)
        return None

    def __call__(self, input=None, batch_size=1):
        """execute model predict

        Returns:
            dict: the prediction results
        """
        results = self.predict()

    def predict(self):
        """predict using specified model
        """
        # self.update_config()
        result = self.pdx_model.predict(**self.get_predict_kwargs())
        assert result.returncode == 0, f"Encountered an unexpected error({result.returncode}) in predicting!"

    def get_predict_kwargs(self) -> dict:
        """get key-value arguments of model predict function

        Returns:
            dict: the arguments of predict function.
        """
        return {
            "weight_path": self.predict_config.model_dir,
            "input_path": self.predict_config.input_path,
            "device": self.global_config.device,
            "save_dir": self.global_config.output
        }

    def _get_post_transforms_for_data(self):
        pass

    def _get_pre_transforms_for_data(self):
        pass

    def _run(self):
        pass

    def get_input_keys(self):
        """ get input keys """
        pass

    def get_output_keys(self):
        """ get output keys """
        pass
