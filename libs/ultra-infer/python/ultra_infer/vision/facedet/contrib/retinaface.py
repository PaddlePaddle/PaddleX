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


class RetinaFace(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a RetinaFace model exported by RetinaFace.

        :param model_file: (str)Path of model file, e.g ./retinaface.onnx
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(RetinaFace, self).__init__(runtime_option)

        self._model = C.vision.facedet.RetinaFace(
            model_file, params_file, self._runtime_option, model_format
        )
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "RetinaFace initialize failed."

    def predict(self, input_image, conf_threshold=0.7, nms_iou_threshold=0.3):
        """Detect the location and key points of human faces from an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :param conf_threshold: confidence threashold for postprocessing, default is 0.7
        :param nms_iou_threshold: iou threashold for NMS, default is 0.3
        :return: FaceDetectionResult
        """
        return self._model.predict(input_image, conf_threshold, nms_iou_threshold)

    # 一些跟模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [640, 480]改变预处理时resize的大小（前提是模型支持）
    @property
    def size(self):
        """
        Argument for image preprocessing step, the preprocess image size, tuple of (width, height), default (640, 640)
        """
        return self._model.size

    @property
    def variance(self):
        """
        Argument for image postprocessing step, variance in RetinaFace's prior-box(anchor) generate process, default (0.1, 0.2)
        """
        return self._model.variance

    @property
    def downsample_strides(self):
        """
        Argument for image postprocessing step, downsample strides (namely, steps) for RetinaFace to generate anchors, will take (8,16,32) as default values
        """
        return self._model.downsample_strides

    @property
    def min_sizes(self):
        """
        Argument for image postprocessing step, min sizes, width and height for each anchor, default min_sizes = [[16, 32], [64, 128], [256, 512]]
        """
        return self._model.min_sizes

    @property
    def landmarks_per_face(self):
        """
        Argument for image postprocessing step, landmarks_per_face, default 5 in RetinaFace
        """
        return self._model.landmarks_per_face

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

    @variance.setter
    def variance(self, value):
        assert isinstance(
            value, (list, tuple)
        ), "The value to set `variance` must be type of tuple or list."
        assert (
            len(value) == 2
        ), "The value to set `variance` must contains 2 elements".format(len(value))
        self._model.variance = value

    @downsample_strides.setter
    def downsample_strides(self, value):
        assert isinstance(
            value, list
        ), "The value to set `downsample_strides` must be type of list."
        self._model.downsample_strides = value

    @min_sizes.setter
    def min_sizes(self, value):
        assert isinstance(
            value, list
        ), "The value to set `min_sizes` must be type of list."
        self._model.min_sizes = value

    @landmarks_per_face.setter
    def landmarks_per_face(self, value):
        assert isinstance(
            value, int
        ), "The value to set `landmarks_per_face` must be type of int."
        self._model.landmarks_per_face = value
