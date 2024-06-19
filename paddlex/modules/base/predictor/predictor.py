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



import os
from copy import deepcopy
from abc import ABC, abstractmethod

from .utils.paddle_inference_predictor import _PaddleInferencePredictor, PaddleInferenceOption
from .utils.mixin import FromDictMixin
from .utils.batch import batchable_method, Batcher
from .utils.node import Node
from .utils.official_models import official_models
from ....utils.device import get_device
from ....utils import logging
from ....utils.config import AttrDict


class BasePredictor(ABC, FromDictMixin, Node):
    """ Base Predictor """
    __is_base = True

    MODEL_FILE_TAG = 'inference'

    def __init__(self,
                 model_dir,
                 kernel_option,
                 pre_transforms=None,
                 post_transforms=None):
        super().__init__()
        self.model_dir = model_dir
        self.pre_transforms = pre_transforms
        self.post_transforms = post_transforms
        self.kernel_option = kernel_option

        param_path = os.path.join(model_dir, f"{self.MODEL_FILE_TAG}.pdiparams")
        model_path = os.path.join(model_dir, f"{self.MODEL_FILE_TAG}.pdmodel")
        self._predictor = _PaddleInferencePredictor(
            param_path=param_path, model_path=model_path, option=kernel_option)

        self.other_src = self.load_other_src()

    def predict(self, input, batch_size=1):
        """ predict """
        if not isinstance(input, dict) and not (isinstance(input, list) and all(
                isinstance(ele, dict) for ele in input)):
            raise TypeError(f"`input` should be a dict or a list of dicts.")

        orig_input = input
        if isinstance(input, dict):
            input = [input]

        logging.info(
            f"Running {self.__class__.__name__}\nModel: {self.model_dir}\nEnv: {self.kernel_option}\n"
        )
        data = input[0]
        if self.pre_transforms is not None:
            pre_tfs = self.pre_transforms
        else:
            pre_tfs = self._get_pre_transforms_for_data(data)
        logging.info(
            f"The following transformation operators will be used for data preprocessing:\n\
{self._format_transforms(pre_tfs)}\n")
        if self.post_transforms is not None:
            post_tfs = self.post_transforms
        else:
            post_tfs = self._get_post_transforms_for_data(data)
        logging.info(
            f"The following transformation operators will be used for postprocessing:\n\
{self._format_transforms(post_tfs)}\n")

        output = []
        for mini_batch in Batcher(input, batch_size=batch_size):
            mini_batch = self._preprocess(mini_batch, pre_transforms=pre_tfs)

            for data in mini_batch:
                self.check_input_keys(data)

            mini_batch = self._run(batch_input=mini_batch)

            for data in mini_batch:
                self.check_output_keys(data)

            mini_batch = self._postprocess(mini_batch, post_transforms=post_tfs)

            output.extend(mini_batch)

        if isinstance(orig_input, dict):
            return output[0]
        else:
            return output

    @abstractmethod
    def _run(self, batch_input):
        raise NotImplementedError

    @abstractmethod
    def _get_pre_transforms_for_data(self, data):
        """ get preprocess transforms """
        raise NotImplementedError

    @abstractmethod
    def _get_post_transforms_for_data(self, data):
        """ get postprocess transforms """
        raise NotImplementedError

    @batchable_method
    def _preprocess(self, data, pre_transforms):
        """ preprocess """
        for tf in pre_transforms:
            data = tf(data)
        return data

    @batchable_method
    def _postprocess(self, data, post_transforms):
        """ postprocess """
        for tf in post_transforms:
            data = tf(data)
        return data

    def _format_transforms(self, transforms):
        """ format transforms """
        lines = ['[']
        for tf in transforms:
            s = '\t'
            s += str(tf)
            lines.append(s)
        lines.append(']')
        return '\n'.join(lines)

    def load_other_src(self):
        """ load other source
        """
        return None


class PredictorBuilderByConfig(object):
    """build model predictor
    """

    def __init__(self, config):
        """
        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        model_name = config.Global.model

        device = config.Global.device.split(':')[0]

        predict_config = deepcopy(config.Predict)
        model_dir = predict_config.pop('model_dir')
        kernel_setting = predict_config.pop('kernel_option', {})
        kernel_setting.setdefault('device', device)
        kernel_option = PaddleInferenceOption(**kernel_setting)

        self.input_path = predict_config.pop('input_path')

        self.predictor = BasePredictor.get(model_name)(model_dir, kernel_option,
                                                       **predict_config)
        self.output = config.Global.output

    def __call__(self):
        data = {
            "input_path": self.input_path,
            "cli_flag": True,
            "output_dir": self.output
        }
        self.predictor.predict(data)


def build_predictor(*args, **kwargs):
    """build predictor by config for dev
    """
    return PredictorBuilderByConfig(*args, **kwargs)


def create_model(model_name,
                 model_dir=None,
                 kernel_option=None,
                 pre_transforms=None,
                 post_transforms=None,
                 *args,
                 **kwargs):
    """create model for predicting using inference model
    """
    kernel_option = PaddleInferenceOption(
    ) if kernel_option is None else kernel_option
    model_dir = official_models[model_name] if model_dir is None else model_dir
    return BasePredictor.get(model_name)(model_dir, kernel_option,
                                         pre_transforms, post_transforms, *args,
                                         **kwargs)