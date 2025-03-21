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


class CaddnPreprocessor:
    def __init__(self, config_file):
        """Create a preprocessor for Caddn"""
        self._preprocessor = C.vision.perception.CaddnPreprocessor(config_file)

    def run(self, input_ims, cam_data, lidar_data):
        """Preprocess input images for Caddn

        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims, cam_data, lidar_data)


class CaddnPostprocessor:
    def __init__(self):
        """Create a postprocessor for Caddn"""
        self._postprocessor = C.vision.perception.CaddnPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for Caddn

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of PerceptionResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class Caddn(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a Caddn model exported by Caddn.

        :param model_file: (str)Path of model file, e.g ./Caddn.pdmodel
        :param params_file: (str)Path of parameters file, e.g ./Caddn.pdiparams
        :param config_file: (str)Path of config file, e.g ./infer_cfg.yaml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(Caddn, self).__init__(runtime_option)

        self._model = C.vision.perception.Caddn(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "Caddn initialize failed."

    def predict(self, input_image, cam_data, lidar_data):
        """Detect an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :param: cam_data: (list)The input camera data
        :param: lidar_data: (list)The input lidar data
        :return: PerceptionResult
        """
        return self._model.predict(input_image, cam_data, lidar_data)

    def batch_predict(self, images, cam_data, lidar_data):
        """Classify a batch of input image

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :param: cam_data: (list)The input camera data
        :param: lidar_data: (list)The input lidar data
        :return list of PerceptionResult
        """

        return self._model.batch_predict(images, cam_data, lidar_data)

    @property
    def preprocessor(self):
        """Get CaddnPreprocessor object of the loaded model

        :return CaddnPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get CaddnPostprocessor object of the loaded model

        :return CaddnPostprocessor
        """
        return self._model.postprocessor
