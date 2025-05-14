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

from __future__ import absolute_import

from .... import UltraInferModel, ModelFormat
from .... import c_lib_wrap as C


class ResNet(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a image classification model exported by torchvision.ResNet.

        :param model_file: (str)Path of model file, e.g resnet/resnet50.onnx
        :param params_file: (str)Path of parameters file, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model, default is ONNX
        """

        # call super() to initialize the backend_option
        # the result of initialization will be saved in self._runtime_option
        super(ResNet, self).__init__(runtime_option)

        self._model = C.vision.classification.ResNet(
            model_file, params_file, self._runtime_option, model_format
        )
        # self.initialized shows the initialization of the model is successful or not

        assert self.initialized, "ResNet initialize failed."

    # Predict and return the inference result of "input_image".
    def predict(self, input_image, topk=1):
        """Classify an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :param topk: (int)The topk result by the classify confidence score, default 1
        :return: ClassifyResult
        """
        return self._model.predict(input_image, topk)

    # Implement the setter and getter method for variables
    @property
    def size(self):
        """
        Returns the preprocess image size, default size = [224, 224];
        """
        return self._model.size

    @property
    def mean_vals(self):
        """
        Returns the mean value of normalization, default mean_vals = [0.485f, 0.456f, 0.406f];
        """
        return self._model.mean_vals

    @property
    def std_vals(self):
        """
        Returns the std value of normalization, default std_vals = [0.229f, 0.224f, 0.225f];
        """
        return self._model.std_vals

    @size.setter
    def size(self, wh):
        assert isinstance(
            wh, (list, tuple)
        ), "The value to set `size` must be type of tuple or list."
        assert (
            len(wh) == 2
        ), "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh)
        )
        self._model.size = wh

    @mean_vals.setter
    def mean_vals(self, value):
        assert isinstance(
            value, list
        ), "The value to set `mean_vals` must be type of list."
        self._model.mean_vals = value

    @std_vals.setter
    def std_vals(self, value):
        assert isinstance(
            value, list
        ), "The value to set `std_vals` must be type of list."
        self._model.std_vals = value
