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
from .... import UltraInferModel, ModelFormat
from .... import c_lib_wrap as C
from ...common import ProcessorManager
from ...detection.ppdet import PicoDet


class PPShiTuV2Detector(PicoDet):
    """Detect main body from an input image."""

    ...


class PPShiTuV2RecognizerPreprocessor(ProcessorManager):
    def __init__(self, config_file):
        """Create a preprocessor for PPShiTuV2Recognizer from configuration file

        :param config_file: (str)Path of configuration file, e.g PPLCNet/inference_cls.yaml
        """
        super(PPShiTuV2RecognizerPreprocessor, self).__init__()
        self._manager = C.vision.classification.PPShiTuV2RecognizerPreprocessor(
            config_file
        )

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._manager.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._manager.disable_permute()

    def initial_resize_on_cpu(self, v):
        """
        When the initial operator is Resize, and input image size is large,
        maybe it's better to run resize on CPU, because the HostToDevice memcpy
        is time consuming. Set this True to run the initial resize on CPU.
        :param: v: True or False
        """
        self._manager.initial_resize_on_cpu(v)


class PPShiTuV2RecognizerPostprocessor:
    def __init__(self, topk=1):
        """Create a postprocessor for PPShiTuV2Recognizer"""
        self._postprocessor = C.vision.classification.PPShiTuV2RecognizerPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for PPShiTuV2Recognizer

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of ClassifyResult, the feature vector is ClassifyResult.feature (If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class PPShiTuV2Recognizer(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a image PPShiTuV2Recognizer model exported by PaddleClas.

        :param model_file: (str)Path of model file, e.g PPLCNet/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g PPLCNet/inference.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str) Path of configuration file for deploy, e.g PPLCNet/inference_cls.yaml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPShiTuV2Recognizer, self).__init__(runtime_option)
        self._model = C.vision.classification.PPShiTuV2Recognizer(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPShiTuV2Recognizer model initialize failed."

    def clone(self):
        """Clone PPShiTuV2Recognizer object

        :return: a new PPShiTuV2Recognizer object
        """

        class PPShiTuV2RecognizerCloneModel(PPShiTuV2Recognizer):
            def __init__(self, model):
                self._model = model

        clone_model = PPShiTuV2RecognizerCloneModel(self._model.clone())
        return clone_model

    def predict(self, im):
        """Extract feature from an input image

        :param im: (numpy.ndarray) The input image data, a 3-D array with layout HWC, BGR format
        :return: ClassifyResult
        """

        return self._model.predict(im)

    def batch_predict(self, images):
        """Extract features from a batch of input image

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of ClassifyResult, the feature vector is ClassifyResult.feature
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get PPShiTuV2RecognizerPreprocessor object of the loaded model

        :return PPShiTuV2RecognizerPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PPShiTuV2RecognizerPostprocessor object of the loaded model

        :return PPShiTuV2RecognizerPostprocessor
        """
        return self._model.postprocessor
