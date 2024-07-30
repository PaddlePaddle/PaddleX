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
import tarfile

from typing import Union
from ...utils import logging
from ..base.build_model import build_model
from ..base.predictor import BasePredictor
from ...utils.errors import raise_unsupported_api_error, raise_model_not_found_error
from .model_list import MODELS
from ...utils.download import download
from ...utils.cache import CACHE_DIR


class TSFCPredictor(BasePredictor):
    """ TS Forecast Model Predictor """
    entities = MODELS

    def __init__(self, model_name, model_dir, kernel_option, output):
        """initialize
        """
        model_dir = self._download_from_url(model_dir)
        self.model_dir = self.uncompress_tar_file(model_dir)

        self.device = kernel_option.get_device()
        self.output = output
        config_path = self.get_config_path()
        self.pdx_config, self.pdx_model = build_model(
            model_name, config_path=config_path)

    def uncompress_tar_file(self, model_dir):
        """unpackage the tar file containing training outputs and update weight path
        """
        if tarfile.is_tarfile(model_dir):
            dest_path = Path(model_dir).parent
            with tarfile.open(model_dir, 'r') as tar:
                tar.extractall(path=dest_path)
            return dest_path / "best_accuracy.pdparams/best_model/model.pdparams"
        return model_dir

    def get_config_path(self) -> Union[str, None]:
        """
        get config path

        Returns:
            config_path (str): The path to the config

        """
        if Path(self.model_dir).exists():
            config_path = Path(self.model_dir).parent.parent / "config.yaml"
            if config_path.exists():
                return config_path
            else:
                logging.warning(
                    f"The config file(`{config_path}`) related to model weight file(`{self.model_dir}`) \
is not exist, use default instead.")
        else:
            raise_model_not_found_error(self.model_dir)
        return None

    def _download_from_url(self, in_path):
        if in_path.startswith("http"):
            file_name = Path(in_path).name
            save_path = Path(CACHE_DIR) / "predict_input" / file_name
            download(in_path, save_path, overwrite=True)
            return save_path.as_posix()
        return in_path
    
    def predict(self, input):
        """execute model predict
        """
        # self.update_config()
        input['input_path'] = self._download_from_url(input['input_path'])
        result = self.pdx_model.predict(**input, **self.get_predict_kwargs())
        assert result.returncode == 0, f"Encountered an unexpected error({result.returncode}) in predicting!"
        return result

    def get_predict_kwargs(self) -> dict:
        """get key-value arguments of model predict function

        Returns:
            dict: the arguments of predict function.
        """
        return {
            "weight_path": self.model_dir,
            "device": self.device,
            "save_dir": self.output
        }

    def _get_post_transforms_from_config(self):
        pass

    def _get_pre_transforms_from_config(self):
        pass

    def _run(self):
        pass

    def get_input_keys(self):
        """ get input keys """
        return ["input_path"]

    def get_output_keys(self):
        """ get output keys """
        pass
