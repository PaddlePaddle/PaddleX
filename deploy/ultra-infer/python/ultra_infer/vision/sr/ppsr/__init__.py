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


class PPMSVSR(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a VSR model exported by PaddleGAN.

        :param model_file: (str)Path of model file, e.g PPMSVSR/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g PPMSVSR/inference.pdiparams
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPMSVSR, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PPMSVSR model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.sr.PPMSVSR(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPMSVSR model initialize failed."

    def predict(self, input_images):
        """Predict the super resolution frame sequences for an input frame sequences

        :param input_images: list[numpy.ndarray] The input image data, 3-D array with layout HWC, BGR format
        :return: list[numpy.ndarray]
        """
        assert input_images is not None, "The input image data is None."
        return self._model.predict(input_images)


class EDVR(PPMSVSR):
    def __init__(
        self,
        model_file,
        params_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a EDVR model exported by PaddleGAN.

        :param model_file: (str)Path of model file, e.g EDVR/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g EDVR/inference.pdiparams
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPMSVSR, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "EDVR model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.sr.EDVR(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "EDVR model initialize failed."

    def predict(self, input_images):
        """Predict the super resolution frame sequences for an input frame sequences

        :param input_images: list[numpy.ndarray] The input image data, 3-D array with layout HWC, BGR format
        :return: list[numpy.ndarray]
        """
        assert input_images is not None, "The input image data is None."
        return self._model.predict(input_images)


class BasicVSR(PPMSVSR):
    def __init__(
        self,
        model_file,
        params_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a EDVR model exported by PaddleGAN.

        :param model_file: (str)Path of model file, e.g BasicVSR/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g BasicVSR/inference.pdiparams
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPMSVSR, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "BasicVSR model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.sr.BasicVSR(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "BasicVSR model initialize failed."

    def predict(self, input_images):
        """Predict the super resolution frame sequences for an input frame sequences

        :param input_images: list[numpy.ndarray] The input image data, 3-D array with layout HWC, BGR format
        :return: list[numpy.ndarray]
        """
        assert input_images is not None, "The input image data is None."
        return self._model.predict(input_images)
