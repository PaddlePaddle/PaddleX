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

from .... import ModelFormat, UltraInferModel
from .... import c_lib_wrap as C


class FSANet(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a headpose model exported by FSANet.

        :param model_file: (str)Path of model file, e.g fsanet/fsanet-var.onnx
        :param params_file: (str)Path of parameters file, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model, default is ONNX
        """

        super(FSANet, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.ONNX
        ), "FSANet only support model format of ModelFormat.ONNX now."
        self._model = C.vision.headpose.FSANet(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "FSANet initialize failed."

    def predict(self, input_image):
        """Predict an input image headpose

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: HeadPoseResult
        """

        return self._model.predict(input_image)

    @property
    def size(self):
        """
        Returns the preprocess image size, default (64, 64)
        """
        return self._model.size

    @size.setter
    def size(self, wh):
        """
        Set the preprocess image size, default (64, 64)
        """
        assert isinstance(
            wh, (list, tuple)
        ), "The value to set `size` must be type of tuple or list."
        assert (
            len(wh) == 2
        ), "The value to set `size` must contatins 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh)
        )
        self._model.size = wh
