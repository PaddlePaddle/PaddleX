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

from ..... import ModelFormat, UltraInferModel
from ..... import c_lib_wrap as C


class InsightFaceRecognitionPreprocessor:
    def __init__(self):
        """Create a preprocessor for InsightFaceRecognition Model"""
        self._preprocessor = C.vision.faceid.InsightFaceRecognitionPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for InsightFaceRecognition Model

        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor, include image, scale_factor, im_shape
        """
        return self._preprocessor.run(input_ims)

    @property
    def size(self):
        """
        Argument for image preprocessing step, tuple of (width, height),
        decide the target size after resize, default (112, 112)
        """
        return self._preprocessor.size

    @property
    def alpha(self):
        """
        Argument for image preprocessing step, alpha values for normalization,
        default alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
        """
        return self._preprocessor.alpha

    @property
    def beta(self):
        """
        Argument for image preprocessing step, beta values for normalization,
        default beta = {-1.f, -1.f, -1.f}
        """
        return self._preprocessor.beta

    def disable_normalize(self):
        """
        This function will disable normalize in preprocessing step.
        """
        self._preprocessor.disable_normalize()

    def disable_permute(self):
        """
        This function will disable hwc2chw in preprocessing step.
        """
        self._preprocessor.disable_permute()


class InsightFaceRecognitionPostprocessor:
    def __init__(self):
        """Create a postprocessor for InsightFaceRecognition Model"""
        self._postprocessor = C.vision.faceid.InsightFaceRecognitionPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for PaddleClas Model

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of FaceRecognitionResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)

    @property
    def l2_normalize(self):
        """
        confidence threshold for postprocessing, default is 0.5
        """
        return self._postprocessor.l2_normalize


class InsightFaceRecognitionBase(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a InsightFaceRecognitionBase model exported by PaddleClas.

        :param model_file: (str)Path of model file, e.g InsightFaceRecognitionBase/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g InsightFaceRecognitionBase/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(InsightFaceRecognitionBase, self).__init__(runtime_option)
        self._model = C.vision.faceid.InsightFaceRecognitionBase(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "InsightFaceRecognitionBase model initialize failed."

    def predict(self, im):
        """Detect an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: DetectionResult
        """

        assert im is not None, "The input image data is None."
        return self._model.predict(im)

    def batch_predict(self, images):
        """Detect a batch of input image list

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of DetectionResult
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get InsightFaceRecognitionPreprocessor object of the loaded model

        :return InsightFaceRecognitionPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get InsightFaceRecognitionPostprocessor object of the loaded model

        :return InsightFaceRecognitionPostprocessor
        """
        return self._model.postprocessor


class ArcFace(InsightFaceRecognitionBase):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a ArcFace model exported by PaddleClas.
        :param model_file: (str)Path of model file, e.g ArcFace/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ArcFace/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(InsightFaceRecognitionBase, self).__init__(runtime_option)

        self._model = C.vision.faceid.ArcFace(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "ArcFace model initialize failed."


class CosFace(InsightFaceRecognitionBase):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a CosFace model exported by PaddleClas.
        :param model_file: (str)Path of model file, e.g CosFace/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g CosFace/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(InsightFaceRecognitionBase, self).__init__(runtime_option)

        self._model = C.vision.faceid.CosFace(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "CosFace model initialize failed."


class PartialFC(InsightFaceRecognitionBase):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a PartialFC model exported by PaddleClas.
        :param model_file: (str)Path of model file, e.g PartialFC/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g PartialFC/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(InsightFaceRecognitionBase, self).__init__(runtime_option)

        self._model = C.vision.faceid.PartialFC(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "PartialFC model initialize failed."


class VPL(InsightFaceRecognitionBase):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a VPL model exported by PaddleClas.
        :param model_file: (str)Path of model file, e.g VPL/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g VPL/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(InsightFaceRecognitionBase, self).__init__(runtime_option)

        self._model = C.vision.faceid.VPL(
            model_file, params_file, self._runtime_option, model_format
        )
        assert self.initialized, "VPL model initialize failed."
