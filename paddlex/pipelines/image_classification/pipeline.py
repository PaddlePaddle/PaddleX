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

from ..base import BasePipeline
from ...modules.image_classification.model_list import MODELS
from ...modules import create_model, PaddleInferenceOption
from ...modules.image_classification import transforms as T


class ClsPipeline(BasePipeline):
    """Cls Pipeline
    """
    entities = "image_classification"

    def __init__(
            self,
            model_name=None,
            model_dir=None,
            output="./output",
            kernel_option=None,
            device="gpu",
            **kwargs, ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_dir = model_dir
        self.output = output
        self.device = device
        self.kernel_option = kernel_option
        if self.model_name is not None:
            self.load_model()

    def check_model_name(self):
        """ check that model name is valid
        """
        assert self.model_name in MODELS, f"The model name({self.model_name}) error. Only support: {MODELS}."

    def predict(self, input):
        """predict
        """
        return self.model.predict(input)

    def load_model(self):
        """load model predictor
        """
        self.check_model_name()
        kernel_option = self.get_kernel_option(
        ) if self.kernel_option is None else self.kernel_option
        self.model = create_model(
            model_name=self.model_name,
            model_dir=self.model_dir,
            output=self.output,
            kernel_option=kernel_option,
            disable_print=self.disable_print,
            disable_save=self.disable_save, )

    def get_kernel_option(self):
        """get kernel option
        """
        kernel_option = PaddleInferenceOption()
        kernel_option.set_device(self.device)
        return kernel_option

    def update_model(self, model_name_list, model_dir_list):
        """update model

        Args:
            model_name_list (list): list of model name.
            model_dir_list (list): list of model directory.
        """
        assert len(model_name_list) == 1
        self.model_name = model_name_list[0]
        if model_dir_list:
            assert len(model_dir_list) == 1
            self.model_dir = model_dir_list[0]

    def get_input_keys(self):
        """get dict keys of input argument input
        """
        return self.model.get_input_keys()
