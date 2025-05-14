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


class YOLOv5ClsPreprocessor:
    def __init__(self):
        """Create a preprocessor for YOLOv5Cls"""
        self._preprocessor = C.vision.classification.YOLOv5ClsPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for YOLOv5Cls

        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)

    @property
    def size(self):
        """
        Argument for image preprocessing step, the preprocess image size, tuple of (width, height), default size = [224, 224]
        """
        return self._preprocessor.size

    @size.setter
    def size(self, wh):
        assert isinstance(
            wh, (list, tuple)
        ), "The value to set `size` must be type of tuple or list."
        assert (
            len(wh) == 2
        ), "The value to set `size` must contains 2 elements means [width, height], but now it contains {} elements.".format(
            len(wh)
        )
        self._preprocessor.size = wh


class YOLOv5ClsPostprocessor:
    def __init__(self):
        """Create a postprocessor for YOLOv5Cls"""
        self._postprocessor = C.vision.classification.YOLOv5ClsPostprocessor()

    def run(self, runtime_results, ims_info):
        """Postprocess the runtime results for YOLOv5Cls

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :param: ims_info: (list of dict)Record input_shape and output_shape
        :return: list of ClassifyResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, ims_info)

    @property
    def topk(self):
        """
        topk for postprocessing, default is 1
        """
        return self._postprocessor.topk

    @topk.setter
    def topk(self, topk):
        assert isinstance(topk, int), "The value to set `top k` must be type of int."
        self._postprocessor.topk = topk


class YOLOv5Cls(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a YOLOv5Cls model exported by YOLOv5Cls.

        :param model_file: (str)Path of model file, e.g ./YOLOv5Cls.onnx
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(YOLOv5Cls, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.ONNX
        ), "YOLOv5Cls only support model format of ModelFormat.ONNX now."
        self._model = C.vision.classification.YOLOv5Cls(
            model_file, params_file, self._runtime_option, model_format
        )

        assert self.initialized, "YOLOv5Cls initialize failed."

    def predict(self, input_image):
        """Classify an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: ClassifyResult
        """
        assert input_image is not None, "Input image is None."
        return self._model.predict(input_image)

    def batch_predict(self, images):
        """Classify a batch of input image

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of ClassifyResult
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get YOLOv5ClsPreprocessor object of the loaded model

        :return YOLOv5ClsPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get YOLOv5ClsPostprocessor object of the loaded model

        :return YOLOv5ClsPostprocessor
        """
        return self._model.postprocessor
