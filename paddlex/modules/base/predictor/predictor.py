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

from .kernel_option import PaddleInferenceOption
from .utils.paddle_inference_predictor import _PaddleInferencePredictor
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

    MODEL_FILE_TAG = "inference"

    def __init__(
            self,
            model_name,
            model_dir,
            kernel_option,
            output,
            pre_transforms=None,
            post_transforms=None,
            disable_print=False,
            disable_save=False, ):
        super().__init__()
        self.model_name = model_name
        self.model_dir = model_dir
        self.kernel_option = kernel_option
        self.output = output
        self.disable_print = disable_print
        self.disable_save = disable_save
        self.other_src = self.load_other_src()

        logging.debug(
            f"-------------------- {self.__class__.__name__} --------------------\n\
Model: {self.model_dir}\n\
Env: {self.kernel_option}")
        self.pre_tfs, self.post_tfs = self.build_transforms(pre_transforms,
                                                            post_transforms)

        param_path = os.path.join(model_dir, f"{self.MODEL_FILE_TAG}.pdiparams")
        model_path = os.path.join(model_dir, f"{self.MODEL_FILE_TAG}.pdmodel")
        self._predictor = _PaddleInferencePredictor(
            param_path=param_path, model_path=model_path, option=kernel_option)

    def build_transforms(self, pre_transforms, post_transforms):
        """ build pre-transforms and post-transforms
        """
        pre_tfs = pre_transforms if pre_transforms is not None else self._get_pre_transforms_from_config(
        )
        logging.debug(f"Preprocess Ops: {self._format_transforms(pre_tfs)}")
        post_tfs = post_transforms if post_transforms is not None else self._get_post_transforms_from_config(
        )
        logging.debug(f"Postprocessing: {self._format_transforms(post_tfs)}")
        return pre_tfs, post_tfs

    def predict(self, input, batch_size=1):
        """ predict """
        if not isinstance(input, dict) and not (isinstance(input, list) and all(
                isinstance(ele, dict) for ele in input)):
            raise TypeError(f"`input` should be a dict or a list of dicts.")

        orig_input = input
        if isinstance(input, dict):
            input = [input]

        output = []
        for mini_batch in Batcher(input, batch_size=batch_size):
            mini_batch = self._preprocess(
                mini_batch, pre_transforms=self.pre_tfs)

            for data in mini_batch:
                self.check_input_keys(data)

            mini_batch = self._run(batch_input=mini_batch)

            for data in mini_batch:
                self.check_output_keys(data)

            mini_batch = self._postprocess(
                mini_batch, post_transforms=self.post_tfs)

            output.extend(mini_batch)

        if isinstance(orig_input, dict):
            return output[0]
        else:
            return output

    @abstractmethod
    def _run(self, batch_input):
        raise NotImplementedError

    @abstractmethod
    def _get_pre_transforms_from_config(self):
        """ get preprocess transforms """
        raise NotImplementedError

    @abstractmethod
    def _get_post_transforms_from_config(self):
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
        ops_str = ",  ".join([str(tf) for tf in transforms])
        return f"[{ops_str}]"

    def load_other_src(self):
        """ load other source
        """
        return None

    def get_input_keys(self):
        """get keys of input dict
        """
        return self.pre_tfs[0].get_input_keys()


class PredictorBuilderByConfig(object):
    """build model predictor
    """

    def __init__(self, config):
        """
        Args:
            config (AttrDict): PaddleX pipeline config, which is loaded from pipeline yaml file.
        """
        model_name = config.Global.model

        device = config.Global.device

        predict_config = deepcopy(config.Predict)
        model_dir = predict_config.pop('model_dir')
        kernel_setting = predict_config.pop('kernel_option', {})
        kernel_setting.setdefault('device', device)
        kernel_option = PaddleInferenceOption(**kernel_setting)

        self.input_path = predict_config.pop('input_path')

        self.predictor = BasePredictor.get(model_name)(
            model_name=model_name,
            model_dir=model_dir,
            kernel_option=kernel_option,
            output=config.Global.output,
            **predict_config)

    def predict(self):
        """predict
        """
        self.predictor.predict({'input_path': self.input_path})


def build_predictor(*args, **kwargs):
    """build predictor by config for dev
    """
    return PredictorBuilderByConfig(*args, **kwargs)


def create_model(model_name,
                 model_dir=None,
                 kernel_option=None,
                 output="./",
                 pre_transforms=None,
                 post_transforms=None,
                 *args,
                 **kwargs):
    """create model for predicting using inference model
    """
    kernel_option = PaddleInferenceOption(
    ) if kernel_option is None else kernel_option
    if model_dir is None:
        if model_name in official_models:
            model_dir = official_models[model_name]
    return BasePredictor.get(model_name)(model_name=model_name,
                                         model_dir=model_dir,
                                         kernel_option=kernel_option,
                                         output=output,
                                         pre_transforms=pre_transforms,
                                         post_transforms=post_transforms,
                                         *args,
                                         **kwargs)
