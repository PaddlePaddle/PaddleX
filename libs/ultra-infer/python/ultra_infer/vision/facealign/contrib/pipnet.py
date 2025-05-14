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


class PIPNet(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a face alignment model exported by PIPNet.

        :param model_file: (str)Path of model file, e.g ./PIPNet.onnx
        :param params_file: (str)Path of parameters file, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model, default is ONNX
        """

        super(PIPNet, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.ONNX
        ), "PIPNet only support model format of ModelFormat.ONNX now."
        self._model = C.vision.facealign.PIPNet(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "PIPNet initialize failed."

    def predict(self, input_image):
        """Detect an input image landmarks

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: FaceAlignmentResult
        """

        return self._model.predict(input_image)

    @property
    def size(self):
        """
        Returns the preprocess image size, default (256, 256)
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

    @property
    def num_landmarks(self):
        """
        Returns the number of landmarks
        """
        return self._model.num_landmarks

    @size.setter
    def size(self, wh):
        """
        Set the preprocess image size, default (256, 256)
        """
        assert isinstance(
            wh, (list, tuple)
        ), "The value to set `size` must be type of tuple or list."
        assert (
            len(wh) == 2
        ), "The value to set `size` must contains 2 elements means [width, height], but now it contains {} elements.".format(
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

    @num_landmarks.setter
    def num_landmarks(self, value):
        assert isinstance(
            value, int
        ), "The value to set `std_vals` must be type of int."
        self._model.num_landmarks = value
