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


class MODNet(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.ONNX,
    ):
        """Load a MODNet model exported by MODNet.

        :param model_file: (str)Path of model file, e.g ./modnet.onnx
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        # 调用基函数进行backend_option的初始化
        # 初始化后的option保存在self._runtime_option
        super(MODNet, self).__init__(runtime_option)

        self._model = C.vision.matting.MODNet(
            model_file, params_file, self._runtime_option, model_format
        )
        # 通过self.initialized判断整个模型的初始化是否成功
        assert self.initialized, "MODNet initialize failed."

    def predict(self, input_image):
        """Predict the matting result for an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: MattingResult
        """
        return self._model.predict(input_image)

    # 一些跟模型有关的属性封装
    # 多数是预处理相关，可通过修改如model.size = [256, 256]改变预处理时resize的大小（前提是模型支持）
    @property
    def size(self):
        """
        Argument for image preprocessing step, the preprocess image size, tuple of (width, height), default size = [256,256]
        """
        return self._model.size

    @property
    def alpha(self):
        """
        Argument for image preprocessing step, alpha value for normalization, default alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f}
        """
        return self._model.alpha

    @property
    def beta(self):
        """
        Argument for image preprocessing step, beta value for normalization, default beta = {-1.f, -1.f, -1.f}
        """
        return self._model.beta

    @property
    def swap_rb(self):
        """
        Argument for image preprocessing step, whether to swap the B and R channel, such as BGR->RGB, default True.
        """
        return self._model.swap_rb

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

    @alpha.setter
    def alpha(self, value):
        assert isinstance(
            value, (list, tuple)
        ), "The value to set `alpha` must be type of tuple or list."
        assert (
            len(value) == 3
        ), "The value to set `alpha` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value)
        )
        self._model.alpha = value

    @beta.setter
    def beta(self, value):
        assert isinstance(
            value, (list, tuple)
        ), "The value to set `beta` must be type of tuple or list."
        assert (
            len(value) == 3
        ), "The value to set `beta` must contatins 3 elements for each channels, but now it contains {} elements.".format(
            len(value)
        )
        self._model.beta = value

    @swap_rb.setter
    def swap_rb(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `swap_rb` must be type of bool."
        self._model.swap_rb = value
