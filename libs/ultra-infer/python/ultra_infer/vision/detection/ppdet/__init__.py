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
from typing import List, Union

from .... import ModelFormat, UltraInferModel
from .... import c_lib_wrap as C
from ...common import ProcessorManager


class PaddleDetPreprocessor(ProcessorManager):
    def __init__(self, config_file):
        """Create a preprocessor for PaddleDetection Model from configuration file

        :param config_file: (str)Path of configuration file, e.g ppyoloe/infer_cfg.yml
        """
        self._manager = C.vision.detection.PaddleDetPreprocessor(config_file)

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


class NMSOption:
    def __init__(self):
        self.nms_option = C.vision.detection.NMSOption()

    @property
    def background_label(self):
        return self.nms_option.background_label


class NMSRotatedOption:
    def __init__(self):
        self.nms_rotated_option = C.vision.detection.NMSRotatedOption()

    @property
    def background_label(self):
        return self.nms_rotated_option.background_label


class PaddleDetPostprocessor:
    def __init__(self):
        """Create a postprocessor for PaddleDetection Model"""
        self._postprocessor = C.vision.detection.PaddleDetPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for PaddleDetection Model

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of ClassifyResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)

    def apply_nms(self):
        self._postprocessor.apply_nms()

    def set_nms_option(self, nms_option=None):
        """This function will enable decode and nms in postprocess step."""
        if nms_option is None:
            nms_option = NMSOption()
        self._postprocessor.set_nms_option(self, nms_option.nms_option)

    def set_nms_rotated_option(self, nms_rotated_option=None):
        """This function will enable decode and rotated nms in postprocess step."""
        if nms_rotated_option is None:
            nms_rotated_option = NMSRotatedOption()
        self._postprocessor.set_nms_rotated_option(
            self, nms_rotated_option.nms_rotated_option
        )


class PPYOLOE(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPYOLOE model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyoloe/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloe/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PPYOLOE(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPYOLOE model initialize failed."

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

    def clone(self):
        """Clone PPYOLOE object

        :return: a new PPYOLOE object
        """

        class PPYOLOEClone(PPYOLOE):
            def __init__(self, model):
                self._model = model

        clone_model = PPYOLOEClone(self._model.clone())
        return clone_model

    @property
    def preprocessor(self):
        """Get PaddleDetPreprocessor object of the loaded model

        :return PaddleDetPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleDetPostprocessor object of the loaded model

        :return PaddleDetPostprocessor
        """
        return self._model.postprocessor


class PPYOLO(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPYOLO model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyolo/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyolo/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PPYOLO model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PPYOLO(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPYOLO model initialize failed."

    def clone(self):
        """Clone PPYOLO object

        :return: a new PPYOLO object
        """

        class PPYOLOClone(PPYOLO):
            def __init__(self, model):
                self._model = model

        clone_model = PPYOLOClone(self._model.clone())
        return clone_model


class PaddleYOLOX(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOX model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolox/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolox/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PaddleYOLOX model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PaddleYOLOX(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleYOLOX model initialize failed."

    def clone(self):
        """Clone PaddleYOLOX object

        :return: a new PaddleYOLOX object
        """

        class PaddleYOLOXClone(PaddleYOLOX):
            def __init__(self, model):
                self._model = model

        clone_model = PaddleYOLOXClone(self._model.clone())
        return clone_model


class PicoDet(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PicoDet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g picodet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g picodet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PicoDet(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PicoDet model initialize failed."

    def clone(self):
        """Clone PicoDet object

        :return: a new PicoDet object
        """

        class PicoDetClone(PicoDet):
            def __init__(self, model):
                self._model = model

        clone_model = PicoDetClone(self._model.clone())
        return clone_model


class FasterRCNN(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a FasterRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g fasterrcnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g fasterrcnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "FasterRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.FasterRCNN(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "FasterRCNN model initialize failed."

    def clone(self):
        """Clone FasterRCNN object

        :return: a new FasterRCNN object
        """

        class FasterRCNNClone(FasterRCNN):
            def __init__(self, model):
                self._model = model

        clone_model = FasterRCNNClone(self._model.clone())
        return clone_model


class YOLOv3(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOv3 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov3/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov3/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "YOLOv3 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.YOLOv3(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "YOLOv3 model initialize failed."

    def clone(self):
        """Clone YOLOv3 object

        :return: a new YOLOv3 object
        """

        class YOLOv3Clone(YOLOv3):
            def __init__(self, model):
                self._model = model

        clone_model = YOLOv3Clone(self._model.clone())
        return clone_model


class SOLOv2(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a SOLOv2 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g solov2/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g solov2/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g solov2/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "SOLOv2 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.SOLOv2(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "SOLOv2 model initialize failed."

    def clone(self):
        """Clone SOLOv2 object

        :return: a new SOLOv2 object
        """

        class SOLOv2Clone(SOLOv2):
            def __init__(self, model):
                self._model = model

        clone_model = SOLOv2Clone(self._model.clone())
        return clone_model


class MaskRCNN(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a MaskRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g fasterrcnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g fasterrcnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "MaskRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.MaskRCNN(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "MaskRCNN model initialize failed."

    def batch_predict(self, images):
        """Detect a batch of input image list, batch_predict is not supported for maskrcnn now.

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of DetectionResult
        """

        raise Exception("batch_predict is not supported for MaskRCNN model now.")

    def clone(self):
        """Clone MaskRCNN object

        :return: a new MaskRCNN object
        """

        class MaskRCNNClone(MaskRCNN):
            def __init__(self, model):
                self._model = model

        clone_model = MaskRCNNClone(self._model.clone())
        return clone_model


class SSD(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a SSD model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ssd/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ssd/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "SSD model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.SSD(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "SSD model initialize failed."

    def clone(self):
        """Clone SSD object

        :return: a new SSD object
        """

        class SSDClone(SSD):
            def __init__(self, model):
                self._model = model

        clone_model = SSDClone(self._model.clone())
        return clone_model


class PaddleYOLOv5(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOv5 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov5/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov5/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PaddleYOLOv5 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PaddleYOLOv5(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleYOLOv5 model initialize failed."


class PaddleYOLOv6(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOv6 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov6/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov6/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PaddleYOLOv6 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PaddleYOLOv6(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleYOLOv6 model initialize failed."


class PaddleYOLOv7(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOv7 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov7/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov7/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PaddleYOLOv7 model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PaddleYOLOv7(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleYOLOv7 model initialize failed."


class PaddleYOLOv8(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a YOLOv8 model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g yolov8/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g yolov8/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g yolov8/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PaddleYOLOv8(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleYOLOv8 model initialize failed."


class RTMDet(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a RTMDet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g rtmdet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g rtmdet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "RTMDet model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.RTMDet(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "RTMDet model initialize failed."


class CascadeRCNN(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a CascadeRCNN model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g cascadercnn/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g cascadercnn/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "CascadeRCNN model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.CascadeRCNN(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "CascadeRCNN model initialize failed."


class PSSDet(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PSSDet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g pssdet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g pssdet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PSSDet model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PSSDet(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PSSDet model initialize failed."


class RetinaNet(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a RetinaNet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g retinanet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g retinanet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "RetinaNet model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.RetinaNet(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "RetinaNet model initialize failed."


class PPYOLOESOD(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPYOLOESOD model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyoloesod/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloesod/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "PPYOLOESOD model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.PPYOLOESOD(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PPYOLOESOD model initialize failed."


class FCOS(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a FCOS model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g fcos/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g fcos/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "FCOS model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.FCOS(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "FCOS model initialize failed."


class TTFNet(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a TTFNet model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ttfnet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ttfnet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "TTFNet model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.TTFNet(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "TTFNet model initialize failed."


class TOOD(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a TOOD model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g tood/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g tood/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "TOOD model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.TOOD(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "TOOD model initialize failed."


class GFL(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a GFL model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g gfl/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g gfl/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        assert (
            model_format == ModelFormat.PADDLE
        ), "GFL model only support model format of ModelFormat.Paddle now."
        self._model = C.vision.detection.GFL(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "GFL model initialize failed."


class PaddleDetectionModel(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PaddleDetectionModel model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyoloe/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloe/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PaddleDetectionModel, self).__init__(runtime_option)

        self._model = C.vision.detection.PaddleDetectionModel(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleDetectionModel model initialize failed."

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

    def clone(self):
        """Clone PPYOLOE object

        :return: a new PPYOLOE object
        """

        class PPYOLOEClone(PPYOLOE):
            def __init__(self, model):
                self._model = model

        clone_model = PPYOLOEClone(self._model.clone())
        return clone_model

    @property
    def preprocessor(self):
        """Get PaddleDetPreprocessor object of the loaded model

        :return PaddleDetPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleDetPostprocessor object of the loaded model

        :return PaddleDetPostprocessor
        """
        return self._model.postprocessor


class PPYOLOER(PPYOLOE):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a PPYOLOER model exported by PaddleDetection.

        :param model_file: (str)Path of model file, e.g ppyoloe_r/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g ppyoloe_r/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str)Path of configuration file for deployment, e.g ppyoloe_r/infer_cfg.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PPYOLOE, self).__init__(runtime_option)

        self._model = C.vision.detection.PPYOLOER(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PicoDet model initialize failed."

    def clone(self):
        """Clone PPYOLOER object

        :return: a new PPYOLOER object
        """

        class PPYOLOERClone(PPYOLOER):
            def __init__(self, model):
                self._model = model

        clone_model = PPYOLOERClone(self._model.clone())
        return clone_model
