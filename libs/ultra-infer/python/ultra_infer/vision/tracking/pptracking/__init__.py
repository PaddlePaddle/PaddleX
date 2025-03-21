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


class PPTracking(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPTracking model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g pptracking/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloe/model.pdiparams
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPTracking, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PPTracking model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.tracking.PPTracking(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPTracking model initialize failed."

    def predict(self, input_image):
        """Predict the MOT result for an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: MOTResult
        """
        assert input_image is not None, "The input image data is None."
        return self._model.predict(input_image)

    def bind_recorder(self, val):
        """Binding tracking trail

        :param val: (TrailRecorder) trail recorder, which is contained object's id and center point sequence
        :return: None
        """
        self._model.bind_recorder(val)

    def unbind_recorder(self):
        """cancel binding of tracking trail

        :return:
        """
        self._model.unbind_recorder()
