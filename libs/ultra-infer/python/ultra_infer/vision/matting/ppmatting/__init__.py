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

import logging

from .... import ModelFormat, UltraInferModel
from .... import c_lib_wrap as C


class PPMatting(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPMatting model exported by PaddleSeg.

        :param model_file: (str)Path of model file, e.g PPMatting-512/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g PPMatting-512/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g PPMatting-512/deploy.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPMatting, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PPMatting model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.matting.PPMatting(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPMatting model initialize failed."

    def predict(self, input_image):
        """Predict the matting result for an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: MattingResult
        """
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)
