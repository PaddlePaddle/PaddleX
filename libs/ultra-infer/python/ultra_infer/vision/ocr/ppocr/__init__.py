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
import math
import os
import re
import tempfile
from dataclasses import dataclass

from tokenizers import Tokenizer as TokenizerFast

from .... import ModelFormat, UltraInferModel
from .... import c_lib_wrap as C
from ....py_only import PyOnlyProcessorChain
from ....py_only.vision import PyOnlyVisionModel
from ....py_only.vision import processors as P
from ....utils.misc import load_config
from ...common import ProcessorManager
from .utils.ser_vi_layoutxlm.operators import *
from .utils.ser_vi_layoutxlm.transforms import *
from .utils.ser_vi_layoutxlm.vqa_utils import *


def sort_boxes(boxes):
    return C.vision.ocr.sort_boxes(boxes)


class UVDocPreprocessor(ProcessorManager):
    def __init__(self):
        """Create a preprocessor for UVDoc Model"""
        super(UVDocPreprocessor, self).__init__()
        self._manager = C.vision.ocr.UVDocPreprocessor()

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

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


class UVDocPostprocessor:
    def __init__(self):
        """Create a postprocessor for UVDoc Model"""
        self._postprocessor = C.vision.ocr.UVDocPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for UVDoc Model
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class UVDocWarpper(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load OCR recognition model provided by PaddleOCR

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(UVDocWarpper, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.UVDocWarpper()
            self._runnable = False
        else:
            self._model = C.vision.ocr.UVDocWarpper(
                model_file, params_file, self._runtime_option, model_format
            )
            assert self.initialized, "UVDocWarpper initialize failed."
            self._runnable = True

    def clone(self):
        """Clone OCR recognition model object
        :return: a new OCR recognition model object
        """

        class UVDocWarpperClone(UVDocWarpper):
            def __init__(self, model):
                self._model = model

        clone_model = UVDocWarpperClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: rec_text, rec_score
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of rec_text, list of rec_score
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value


class DBDetectorPreprocessor(ProcessorManager):
    def __init__(self):
        """
        Create a preprocessor for DBDetectorModel
        """
        super(DBDetectorPreprocessor, self).__init__()
        self._manager = C.vision.ocr.DBDetectorPreprocessor()

    @property
    def max_side_len(self):
        """Get max_side_len value."""
        return self._manager.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        """Set max_side_len value.
        :param: value: (int) max_side_len value
        """
        assert isinstance(
            value, int
        ), "The value to set `max_side_len` must be type of int."
        self._manager.max_side_len = value

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def static_shape_infer(self):
        return self._manager.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `static_shape_infer` must be type of bool."
        self._manager.static_shape_infer = value

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


class DBDetectorPostprocessor:
    def __init__(self):
        """
        Create a postprocessor for DBDetectorModel
        """
        self._postprocessor = C.vision.ocr.DBDetectorPostprocessor()

    def run(self, runtime_results, batch_det_img_info):
        """Postprocess the runtime results for DBDetectorModel

        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :param: batch_det_img_info: (list of std::array<int, 4>)The output of det_preprocessor
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, batch_det_img_info)

    @property
    def det_db_thresh(self):
        """
        Return the det_db_thresh of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        """Set the det_db_thresh for DBDetectorPostprocessor

        :param: value : the det_db_thresh value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_thresh` must be type of float."
        self._postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        """
        Return the det_db_box_thresh of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        """Set the det_db_box_thresh for DBDetectorPostprocessor

        :param: value : the det_db_box_thresh value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        """
        Return the det_db_unclip_ratio of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        """Set the det_db_unclip_ratio for DBDetectorPostprocessor

        :param: value : the det_db_unclip_ratio value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_score_mode(self):
        """
        Return the det_db_score_mode of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_score_mode

    @property
    def det_db_box_type(self):
        """
        Return the det_db_score_mode of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_box_type

    @det_db_box_type.setter
    def det_db_box_type(self, value):
        """Set the det_db_score_mode for DBDetectorPostprocessor

        :param: value : the det_db_score_mode value
        """
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_box_type = value

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        """Set the det_db_score_mode for DBDetectorPostprocessor

        :param: value : the det_db_score_mode value
        """
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        """
        Return the use_dilation of DBDetectorPostprocessor
        """
        return self._postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        """Set the use_dilation for DBDetectorPostprocessor

        :param: value : the use_dilation value
        """
        assert isinstance(
            value, bool
        ), "The value to set `use_dilation` must be type of bool."
        self._postprocessor.use_dilation = value


class DBCURVEDetectorPostprocessor:
    def __init__(self):
        """
        Create a postprocessor for DBDetectorModel
        """
        self._postprocessor = C.vision.ocr.DBCURVEDetectorPostprocessor()

    def run(self, runtime_results, batch_det_img_info):
        """Postprocess the runtime results for DBDetectorModel

        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :param: batch_det_img_info: (list of std::array<int, 4>)The output of det_preprocessor
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, batch_det_img_info)

    @property
    def det_db_thresh(self):
        """
        Return the det_db_thresh of DBCURVEDetectorPostprocessor
        """
        return self._postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        """Set the det_db_thresh for DBCURVEDetectorPostprocessor

        :param: value : the det_db_thresh value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_thresh` must be type of float."
        self._postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        """
        Return the det_db_box_thresh of DBCURVEDetectorPostprocessor
        """
        return self._postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        """Set the det_db_box_thresh for DBCURVEDetectorPostprocessor

        :param: value : the det_db_box_thresh value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        """
        Return the det_db_unclip_ratio of DBCURVEDetectorPostprocessor
        """
        return self._postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        """Set the det_db_unclip_ratio for DBCURVEDetectorPostprocessor

        :param: value : the det_db_unclip_ratio value
        """
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_score_mode(self):
        """
        Return the det_db_score_mode of DBCURVEDetectorPostprocessor
        """
        return self._postprocessor.det_db_score_mode

    @property
    def det_db_box_type(self):
        """
        Return the det_db_score_mode of DBDetectorPostprocessor
        """
        return self._postprocessor.det_db_box_type

    @det_db_box_type.setter
    def det_db_box_type(self, value):
        """Set the det_db_score_mode for DBDetectorPostprocessor

        :param: value : the det_db_score_mode value
        """
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_box_type = value

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        """Set the det_db_score_mode for DBDetectorPostprocessor

        :param: value : the det_db_score_mode value
        """
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        """
        Return the use_dilation of DBDetectorPostprocessor
        """
        return self._postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        """Set the use_dilation for DBDetectorPostprocessor

        :param: value : the use_dilation value
        """
        assert isinstance(
            value, bool
        ), "The value to set `use_dilation` must be type of bool."
        self._postprocessor.use_dilation = value


class DBDetector(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load OCR detection model provided by PaddleOCR.

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(DBDetector, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.DBDetector()
            self._runnable = False
        else:
            self._model = C.vision.ocr.DBDetector(
                model_file, params_file, self._runtime_option, model_format
            )
            assert self.initialized, "DBDetector initialize failed."
            self._runnable = True

    def clone(self):
        """Clone OCR detection model object

        :return: a new OCR detection model object
        """

        class DBDetectorClone(DBDetector):
            def __init__(self, model):
                self._model = model

        clone_model = DBDetectorClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: boxes
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: batch_boxes
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @property
    def postprocessor(self):
        return self._model.postprocessor

    # Det Preprocessor Property
    @property
    def max_side_len(self):
        return self._model.preprocessor.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int
        ), "The value to set `max_side_len` must be type of int."
        self._model.preprocessor.max_side_len = value

    # Det Ppstprocessor Property
    @property
    def det_db_thresh(self):
        return self._model.postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_thresh` must be type of float."
        self._model.postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        return self._model.postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._model.postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        return self._model.postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._model.postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_box_type(self):
        return self._model.postprocessor.det_db_box_type

    @det_db_box_type.setter
    def det_db_box_type(self, value):
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._model.postprocessor.det_db_box_type = value

    @property
    def det_db_score_mode(self):
        return self._model.postprocessor.det_db_score_mode

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._model.postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        return self._model.postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `use_dilation` must be type of bool."
        self._model.postprocessor.use_dilation = value


class DBCURVEDetector(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load OCR detection model provided by PaddleOCR.

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_det_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_det_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(DBCURVEDetector, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.DBCURVEDetector()
            self._runnable = False
        else:
            self._model = C.vision.ocr.DBCURVEDetector(
                model_file, params_file, self._runtime_option, model_format
            )
            assert self.initialized, "DBCURVEDetector initialize failed."
            self._runnable = True

    def clone(self):
        """Clone OCR detection model object

        :return: a new OCR detection model object
        """

        class DBCURVEDetectorClone(DBCURVEDetector):
            def __init__(self, model):
                self._model = model

        clone_model = DBCURVEDetectorClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: boxes
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: batch_boxes
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @property
    def postprocessor(self):
        return self._model.postprocessor

    # Det Preprocessor Property
    @property
    def max_side_len(self):
        return self._model.preprocessor.max_side_len

    @max_side_len.setter
    def max_side_len(self, value):
        assert isinstance(
            value, int
        ), "The value to set `max_side_len` must be type of int."
        self._model.preprocessor.max_side_len = value

    # Det Ppstprocessor Property
    @property
    def det_db_thresh(self):
        return self._model.postprocessor.det_db_thresh

    @det_db_thresh.setter
    def det_db_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_thresh` must be type of float."
        self._model.postprocessor.det_db_thresh = value

    @property
    def det_db_box_thresh(self):
        return self._model.postprocessor.det_db_box_thresh

    @det_db_box_thresh.setter
    def det_db_box_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_box_thresh` must be type of float."
        self._model.postprocessor.det_db_box_thresh = value

    @property
    def det_db_unclip_ratio(self):
        return self._model.postprocessor.det_db_unclip_ratio

    @det_db_unclip_ratio.setter
    def det_db_unclip_ratio(self, value):
        assert isinstance(
            value, float
        ), "The value to set `det_db_unclip_ratio` must be type of float."
        self._model.postprocessor.det_db_unclip_ratio = value

    @property
    def det_db_box_type(self):
        return self._model.postprocessor.det_db_box_type

    @det_db_box_type.setter
    def det_db_box_type(self, value):
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._model.postprocessor.det_db_box_type = value

    @property
    def det_db_score_mode(self):
        return self._model.postprocessor.det_db_score_mode

    @det_db_score_mode.setter
    def det_db_score_mode(self, value):
        assert isinstance(
            value, str
        ), "The value to set `det_db_score_mode` must be type of str."
        self._model.postprocessor.det_db_score_mode = value

    @property
    def use_dilation(self):
        return self._model.postprocessor.use_dilation

    @use_dilation.setter
    def use_dilation(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `use_dilation` must be type of bool."
        self._model.postprocessor.use_dilation = value


class ClassifierPreprocessor(ProcessorManager):
    def __init__(self):
        """Create a preprocessor for ClassifierModel"""
        super(ClassifierPreprocessor, self).__init__()
        self._manager = C.vision.ocr.ClassifierPreprocessor()

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def cls_image_shape(self):
        return self._manager.cls_image_shape

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value, list
        ), "The value to set `cls_image_shape` must be type of list."
        self._manager.cls_image_shape = value

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


class ClassifierPostprocessor:
    def __init__(self):
        """Create a postprocessor for ClassifierModel"""
        self._postprocessor = C.vision.ocr.ClassifierPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for ClassifierModel
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)

    @property
    def cls_thresh(self):
        """
        Return the cls_thresh of ClassifierPostprocessor
        """
        return self._postprocessor.cls_thresh

    @cls_thresh.setter
    def cls_thresh(self, value):
        """Set the cls_thresh for ClassifierPostprocessor

        :param: value: the value of cls_thresh
        """
        assert isinstance(
            value, float
        ), "The value to set `cls_thresh` must be type of float."
        self._postprocessor.cls_thresh = value


class Classifier(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load OCR classification model provided by PaddleOCR.

        :param model_file: (str)Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(Classifier, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.Classifier()
            self._runnable = False
        else:
            self._model = C.vision.ocr.Classifier(
                model_file, params_file, self._runtime_option, model_format
            )
            assert self.initialized, "Classifier initialize failed."
            self._runnable = True

    def clone(self):
        """Clone OCR classification model object
        :return: a new OCR classification model object
        """

        class ClassifierClone(Classifier):
            def __init__(self, model):
                self._model = model

        clone_model = ClassifierClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: cls_label, cls_score
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of cls_label, list of cls_score
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value

    @property
    def cls_image_shape(self):
        return self._model.preprocessor.cls_image_shape

    @cls_image_shape.setter
    def cls_image_shape(self, value):
        assert isinstance(
            value, list
        ), "The value to set `cls_image_shape` must be type of list."
        self._model.preprocessor.cls_image_shape = value

    # Cls Postprocessor Property
    @property
    def cls_thresh(self):
        return self._model.postprocessor.cls_thresh

    @cls_thresh.setter
    def cls_thresh(self, value):
        assert isinstance(
            value, float
        ), "The value to set `cls_thresh` must be type of float."
        self._model.postprocessor.cls_thresh = value


class RecognizerPreprocessor(ProcessorManager):
    def __init__(self):
        """Create a preprocessor for RecognizerModel"""
        super(RecognizerPreprocessor, self).__init__()
        self._manager = C.vision.ocr.RecognizerPreprocessor()

    @property
    def static_shape_infer(self):
        return self._manager.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `static_shape_infer` must be type of bool."
        self._manager.static_shape_infer = value

    def set_normalize(self, mean, std, is_scale):
        """Set preprocess normalize parameters, please call this API to
           customize the normalize parameters, otherwise it will use the default
           normalize parameters.
        :param: mean: (list of float) mean values
        :param: std: (list of float) std values
        :param: is_scale: (boolean) whether to scale
        """
        self._manager.set_normalize(mean, std, is_scale)

    @property
    def rec_image_shape(self):
        return self._manager.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value, list
        ), "The value to set `rec_image_shape` must be type of list."
        self._manager.rec_image_shape = value

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


class RecognizerPostprocessor:
    def __init__(self, label_path):
        """Create a postprocessor for RecognizerModel
        :param label_path: (str)Path of label file
        """
        self._postprocessor = C.vision.ocr.RecognizerPostprocessor(label_path)

    def run(self, runtime_results):
        """Postprocess the runtime results for RecognizerModel
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class Recognizer(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        label_path="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load OCR recognition model provided by PaddleOCR

        :param model_file: (str)Path of model file, e.g ./ch_PP-OCRv3_rec_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_PP-OCRv3_rec_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param label_path: (str)Path of label file used by OCR recognition model. e.g ./ppocr_keys_v1.txt
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(Recognizer, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.Recognizer()
            self._runnable = False
        else:
            self._model = C.vision.ocr.Recognizer(
                model_file, params_file, label_path, self._runtime_option, model_format
            )
            assert self.initialized, "Recognizer initialize failed."
            self._runnable = True

    def clone(self):
        """Clone OCR recognition model object
        :return: a new OCR recognition model object
        """

        class RecognizerClone(Recognizer):
            def __init__(self, model):
                self._model = model

        clone_model = RecognizerClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: rec_text, rec_score
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of rec_text, list of rec_score
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value

    @property
    def static_shape_infer(self):
        return self._model.preprocessor.static_shape_infer

    @static_shape_infer.setter
    def static_shape_infer(self, value):
        assert isinstance(
            value, bool
        ), "The value to set `static_shape_infer` must be type of bool."
        self._model.preprocessor.static_shape_infer = value

    @property
    def rec_image_shape(self):
        return self._model.preprocessor.rec_image_shape

    @rec_image_shape.setter
    def rec_image_shape(self, value):
        assert isinstance(
            value, list
        ), "The value to set `rec_image_shape` must be type of list."
        self._model.preprocessor.rec_image_shape = value


class StructureV2TablePreprocessor:
    def __init__(self):
        """Create a preprocessor for StructureV2Table Model"""
        self._preprocessor = C.vision.ocr.StructureV2TablePreprocessor()

    def run(self, input_ims):
        """Preprocess input images for StructureV2TableModel
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)


class StructureV2TablePostprocessor:
    def __init__(self, dict_path):
        """Create a postprocessor for StructureV2Table Model"""
        self._postprocessor = C.vision.ocr.StructureV2TablePostprocessor(dict_path)

    def run(self, runtime_results):
        """Postprocess the runtime results for StructureV2Table Model
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class StructureV2Table(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        table_char_dict_path="",
        box_shape="ori",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load StructureV2Table model provided by PP-StructureV2.

        :param model_file: (str)Path of model file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ch_ppocr_mobile_v2.0_cls_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param table_char_dict_path: (str)Path of table_char_dict file, e.g ../ppocr/utils/dict/table_structure_dict_ch.txt
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(StructureV2Table, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.StructureV2Table()
            self._runnable = False
        else:
            self._model = C.vision.ocr.StructureV2Table(
                model_file,
                params_file,
                table_char_dict_path,
                box_shape,
                self._runtime_option,
                model_format,
            )
            assert self.initialized, "Classifier initialize failed."
            self._runnable = True

    def clone(self):
        """Clone StructureV2Table model object
        :return: a new StructureV2Table model object
        """

        class StructureV2TableClone(StructureV2Table):
            def __init__(self, model):
                self._model = model

        clone_model = StructureV2TableClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: bbox, structure
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of bbox list, list of structure
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value


class StructureV2LayoutPreprocessor:
    def __init__(self):
        """Create a preprocessor for StructureV2Layout Model"""
        self._preprocessor = C.vision.ocr.StructureV2LayoutPreprocessor()

    def run(self, input_ims):
        """Preprocess input images for StructureV2Layout Model
        :param: input_ims: (list of numpy.ndarray)The input image
        :return: list of FDTensor
        """
        return self._preprocessor.run(input_ims)


class StructureV2LayoutPostprocessor:
    def __init__(self):
        """Create a postprocessor for StructureV2Layout Model"""
        self._postprocessor = C.vision.ocr.StructureV2LayoutPostprocessor()

    def run(self, runtime_results):
        """Postprocess the runtime results for StructureV2Layout Model
        :param: runtime_results: (list of FDTensor or list of pyArray)The output FDTensor results from runtime
        :return: list of Result(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class StructureV2Layout(UltraInferModel):
    def __init__(
        self,
        model_file="",
        params_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load StructureV2Layout model provided by PP-StructureV2.

        :param model_file: (str)Path of model file, e.g ./picodet_lcnet_x1_0_fgd_layout_infer/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./picodet_lcnet_x1_0_fgd_layout_infer/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(StructureV2Layout, self).__init__(runtime_option)

        if len(model_file) == 0:
            self._model = C.vision.ocr.StructureV2Layout()
            self._runnable = False
        else:
            self._model = C.vision.ocr.StructureV2Layout(
                model_file, params_file, self._runtime_option, model_format
            )
            assert self.initialized, "StructureV2Layout model initialize failed."
            self._runnable = True

    def clone(self):
        """Clone StructureV2Layout model object
        :return: a new StructureV2Table model object
        """

        class StructureV2LayoutClone(StructureV2Layout):
            def __init__(self, model):
                self._model = model

        clone_model = StructureV2LayoutClone(self._model.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: bboxes
        """
        if self._runnable:
            return self._model.predict(input_image)
        return False

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of bboxes list
        """
        if self._runnable:
            return self._model.batch_predict(images)
        return False

    @property
    def preprocessor(self):
        return self._model.preprocessor

    @preprocessor.setter
    def preprocessor(self, value):
        self._model.preprocessor = value

    @property
    def postprocessor(self):
        return self._model.postprocessor

    @postprocessor.setter
    def postprocessor(self, value):
        self._model.postprocessor = value


class PPOCRv4(UltraInferModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (UltraInferModel) The detection model object created by ultra_infer.vision.ocr.DBDetector.
        :param cls_model: (UltraInferModel) The classification model object created by ultra_infer.vision.ocr.Classifier.
        :param rec_model: (UltraInferModel) The recognition model object created by ultra_infer.vision.ocr.Recognizer.
        """
        assert (
            det_model is not None and rec_model is not None
        ), "The det_model and rec_model cannot be None."

        self.det_model = det_model
        self.rec_model = rec_model
        self.cls_model = cls_model

        if cls_model is None:
            self.system_ = C.vision.ocr.PPOCRv4(det_model._model, rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv4(
                det_model._model, cls_model._model, rec_model._model
            )

    def clone(self):
        """Clone PPOCRv4 pipeline object
        :return: a new PPOCRv4 pipeline object
        """

        class PPOCRv4Clone(PPOCRv4):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv4Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """
        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value


class PPOCRSystemv4(PPOCRv4):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv4 is deprecated, "
            "please use fd.vision.ocr.PPOCRv4 instead."
        )
        super(PPOCRSystemv4, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv4, self).predict(input_image)


class PPOCRv3(UltraInferModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (UltraInferModel) The detection model object created by ultra_infer.vision.ocr.DBDetector.
        :param cls_model: (UltraInferModel) The classification model object created by ultra_infer.vision.ocr.Classifier.
        :param rec_model: (UltraInferModel) The recognition model object created by ultra_infer.vision.ocr.Recognizer.
        """
        assert (
            det_model is not None and rec_model is not None
        ), "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system_ = C.vision.ocr.PPOCRv3(det_model._model, rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv3(
                det_model._model, cls_model._model, rec_model._model
            )

    def clone(self):
        """Clone PPOCRv3 pipeline object
        :return: a new PPOCRv3 pipeline object
        """

        class PPOCRv3Clone(PPOCRv3):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv3Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image
        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """
        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value


class PPOCRSystemv3(PPOCRv3):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv3 is deprecated, "
            "please use fd.vision.ocr.PPOCRv3 instead."
        )
        super(PPOCRSystemv3, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv3, self).predict(input_image)


class PPOCRv2(UltraInferModel):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        """Consruct a pipeline with text detector, direction classifier and text recognizer models

        :param det_model: (UltraInferModel) The detection model object created by ultra_infer.vision.ocr.DBDetector.
        :param cls_model: (UltraInferModel) The classification model object created by ultra_infer.vision.ocr.Classifier.
        :param rec_model: (UltraInferModel) The recognition model object created by ultra_infer.vision.ocr.Recognizer.
        """
        assert (
            det_model is not None and rec_model is not None
        ), "The det_model and rec_model cannot be None."
        if cls_model is None:
            self.system_ = C.vision.ocr.PPOCRv2(det_model._model, rec_model._model)
        else:
            self.system_ = C.vision.ocr.PPOCRv2(
                det_model._model, cls_model._model, rec_model._model
            )

    def clone(self):
        """Clone PPOCRv3 pipeline object
        :return: a new PPOCRv3 pipeline object
        """

        class PPOCRv2Clone(PPOCRv2):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPOCRv2Clone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """

        return self.system_.batch_predict(images)

    @property
    def cls_batch_size(self):
        return self.system_.cls_batch_size

    @cls_batch_size.setter
    def cls_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `cls_batch_size` must be type of int."
        self.system_.cls_batch_size = value

    @property
    def rec_batch_size(self):
        return self.system_.rec_batch_size

    @rec_batch_size.setter
    def rec_batch_size(self, value):
        assert isinstance(
            value, int
        ), "The value to set `rec_batch_size` must be type of int."
        self.system_.rec_batch_size = value


class PPOCRSystemv2(PPOCRv2):
    def __init__(self, det_model=None, cls_model=None, rec_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPOCRSystemv2 is deprecated, "
            "please use fd.vision.ocr.PPOCRv2 instead."
        )
        super(PPOCRSystemv2, self).__init__(det_model, cls_model, rec_model)

    def predict(self, input_image):
        return super(PPOCRSystemv2, self).predict(input_image)


class PPStructureV2Table(UltraInferModel):
    def __init__(self, det_model=None, rec_model=None, table_model=None):
        """Consruct a pipeline with text detector, text recognizer and table recognizer models

        :param det_model: (UltraInferModel) The detection model object created by ultra_infer.vision.ocr.DBDetector.
        :param rec_model: (UltraInferModel) The recognition model object created by ultra_infer.vision.ocr.Recognizer.
        :param table_model: (UltraInferModel) The table recognition model object created by ultra_infer.vision.ocr.Table.
        """
        assert (
            det_model is not None and rec_model is not None and table_model is not None
        ), "The det_model, rec_model and table_model cannot be None."
        self.system_ = C.vision.ocr.PPStructureV2Table(
            det_model._model,
            rec_model._model,
            table_model._model,
        )

    def clone(self):
        """Clone PPStructureV2Table pipeline object
        :return: a new PPStructureV2Table pipeline object
        """

        class PPStructureV2TableClone(PPStructureV2Table):
            def __init__(self, system):
                self.system_ = system

        clone_model = PPStructureV2TableClone(self.system_.clone())
        return clone_model

    def predict(self, input_image):
        """Predict an input image

        :param input_image: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: OCRResult
        """
        return self.system_.predict(input_image)

    def batch_predict(self, images):
        """Predict a batch of input image
        :param images: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: OCRBatchResult
        """

        return self.system_.batch_predict(images)


class PPStructureV2TableSystem(PPStructureV2Table):
    def __init__(self, det_model=None, rec_model=None, table_model=None):
        logging.warning(
            "DEPRECATED: fd.vision.ocr.PPStructureV2TableSystem is deprecated, "
            "please use fd.vision.ocr.PPStructureV2Table instead."
        )
        super(PPStructureV2TableSystem, self).__init__(
            det_model, rec_model, table_model
        )

    def predict(self, input_image):
        return super(PPStructureV2TableSystem, self).predict(input_image)


class StructureV2SERViLayoutXLMModelPreprocessor:
    def __init__(self, ser_dict_path, use_gpu=True):
        """Create a preprocessor for Ser-Vi-LayoutXLM model.
        :param: ser_dict_path: (str) class file path
        :param: use_gpu: (bool) whether use gpu to OCR process
        """
        self._manager = None
        from paddleocr import PaddleOCR

        self.ocr_engine = PaddleOCR(
            use_angle_cls=False,
            det_model_dir=None,
            rec_model_dir=None,
            show_log=False,
            use_gpu=use_gpu,
        )

        pre_process_list = [
            {
                "VQATokenLabelEncode": {
                    "class_path": ser_dict_path,
                    "contains_re": False,
                    "ocr_engine": self.ocr_engine,
                    "order_method": "tb-yx",
                }
            },
            {"VQATokenPad": {"max_seq_len": 512, "return_attention_mask": True}},
            {"VQASerTokenChunk": {"max_seq_len": 512, "return_attention_mask": True}},
            {"Resize": {"size": [224, 224]}},
            {
                "NormalizeImage": {
                    "std": [58.395, 57.12, 57.375],
                    "mean": [123.675, 116.28, 103.53],
                    "scale": "1",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {
                "KeepKeys": {
                    "keep_keys": [
                        "input_ids",
                        "bbox",
                        "attention_mask",
                        "token_type_ids",
                        "image",
                        "labels",
                        "segment_offset_id",
                        "ocr_info",
                        "entities",
                    ]
                }
            },
        ]

        self.preprocess_op = create_operators(pre_process_list, {"infer_mode": True})

    def _transform(self, data, ops=None):
        """transform"""
        if ops is None:
            ops = []
        for op in ops:
            data = op(data)
            if data is None:
                return None
        return data

    def run(self, input_im):
        """Run preprocess of  Ser-Vi-LayoutXLM model
        :param: input_ims: (numpy.ndarray) input image
        """
        ori_im = input_im.copy()
        data = {"image": input_im}
        data = transform(data, self.preprocess_op)

        for idx in range(len(data)):
            if isinstance(data[idx], np.ndarray):
                data[idx] = np.expand_dims(data[idx], axis=0)
            else:
                data[idx] = [data[idx]]

        return data


class StructureV2SERViLayoutXLMModelPostprocessor:
    def __init__(self, class_path):
        """Create a postprocessor for Ser-Vi-LayoutXLM model.
        :param: class_path: (string) class file path
        """
        self.postprocessor_op = VQASerTokenLayoutLMPostProcess(class_path)

    def run(self, preds, batch=None, *args, **kwargs):
        """Run postprocess of  Ser-Vi-LayoutXLM model.
        :param: preds: (list) results of infering
        """
        return self.postprocessor_op(preds, batch, *args, **kwargs)


class StructureV2SERViLayoutXLMModel(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        ser_dict_path,
        class_path,
        config_file="",
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load SERViLayoutXLM model provided by PP-StructureV2.

        :param model_file: (str)Path of model file, e.g ./ser_vi_layout_xlm/model.pdmodel.
        :param params_file: (str)Path of parameter file, e.g ./ser_vi_layout_xlm/model.pdiparams, if the model format is ONNX, this parameter will be ignored.
        :param ser_dict_path: (str) class file path
        :param class_path: (str) class file path
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU.
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model.
        """
        super(StructureV2SERViLayoutXLMModel, self).__init__(runtime_option)

        assert (
            self._runtime_option.backend != 0
        ), "Runtime Option required backend setting."
        self._model = C.vision.ocr.StructureV2SERViLayoutXLMModel(
            model_file, params_file, config_file, self._runtime_option, model_format
        )

        assert self.initialized, "SERViLayoutXLM model initialize failed."

        self.preprocessor = StructureV2SERViLayoutXLMModelPreprocessor(ser_dict_path)
        self.postprocesser = StructureV2SERViLayoutXLMModelPostprocessor(class_path)

        self.input_name_0 = self._model.get_input_info(0).name
        self.input_name_1 = self._model.get_input_info(1).name
        self.input_name_2 = self._model.get_input_info(2).name
        self.input_name_3 = self._model.get_input_info(3).name

    def predict(self, image):
        assert isinstance(image, np.ndarray), "predict recives numpy.ndarray(BGR)"

        data = self.preprocessor.run(image)
        infer_input = {
            self.input_name_0: data[0],
            self.input_name_1: data[1],
            self.input_name_2: data[2],
            self.input_name_3: data[3],
        }

        infer_result = self._model.infer(infer_input)
        infer_result = infer_result[0]

        post_result = self.postprocesser.run(
            infer_result, segment_offset_ids=data[6], ocr_infos=data[7]
        )

        return post_result

    def batch_predict(self, image_list):
        assert isinstance(image_list, list) and isinstance(
            image_list[0], np.ndarray
        ), "batch_predict recives list of numpy.ndarray(BGR)"

        # reading and preprocessing images
        datas = None
        for image in image_list:
            data = self.preprocessor.run(image)

            # concatenate data to batch
            if datas == None:
                datas = data
            else:
                for idx in range(len(data)):
                    if isinstance(data[idx], np.ndarray):
                        datas[idx] = np.concatenate((datas[idx], data[idx]), axis=0)
                    else:
                        datas[idx].extend(data[idx])

        # infer
        infer_inputs = {
            self.input_name_0: datas[0],
            self.input_name_1: datas[1],
            self.input_name_2: datas[2],
            self.input_name_3: datas[3],
        }

        infer_results = self._model.infer(infer_inputs)
        infer_results = infer_results[0]

        # postprocessing
        post_results = self.postprocesser.run(
            infer_results, segment_offset_ids=datas[6], ocr_infos=datas[7]
        )

        return post_results


class PyOnlyFormulaRecognitionModel(PyOnlyVisionModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        self._model_file = model_file
        self._params_file = params_file
        self._model_format = model_format
        super().__init__(runtime_option)
        self._config = load_config(config_file)
        self._preprocessor = _PyOnlyFormulaRecognitionPreprocessor()
        self._postprocessor = _PyOnlyFormulaRecognitionPostprocessor(
            **self._config["PostProcess"]
        )

    def model_name():
        return "PyOnlyFormulaRecognitionModel"

    def batch_predict(self, imgs):
        data_list = []
        for img in imgs:
            data = {"img": img}
            data = self._preprocessor.run(data)
            data_list.append(data)

        input_name = self._runtime.get_input_info(0).name
        imgs = np.stack([data["img"] for data in data_list], axis=0, dtype=np.float32)
        imgs = np.ascontiguousarray(imgs)
        output_arrs = self._runtime.infer({input_name: imgs})

        results = []
        for score_map in output_arrs[0]:
            data = {"score_map": score_map}
            result = self._postprocessor.run(data)
            results.append(result)
        return results

    def _update_option(self):
        self._option.set_model_path(
            self._model_file, self._params_file, self._model_format
        )


class _PyOnlyFormulaRecognitionPreprocessor(object):
    def __init__(self):
        super().__init__()
        processors = self._build_processors()
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        return self._processor_chain(data)

    def _build_processors(self):
        processors = []
        processors.append(P.LaTeXOCRReisizeNormImg())
        return processors


class _PyOnlyFormulaRecognitionPostprocessor(object):
    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get("name") == "LaTeXOCRDecode":
            self.op = LaTeXOCRDecode(
                character_list=kwargs.get("character_dict"),
            )
        else:
            raise Exception()

    def run(self, data):
        rec_text = self.op.apply(data)
        rec_text = rec_text["rec_text"]
        result = _PyOnlyFormulaRecognitionResult(rec_text=rec_text)
        return result


@dataclass
class _PyOnlyFormulaRecognitionResult(object):
    rec_text: str


class LaTeXOCRDecode(object):
    def __init__(self, character_list=None):
        super().__init__()
        character_list = character_list
        temp_path = tempfile.gettempdir()
        rec_char_dict_path = os.path.join(temp_path, "latexocr_tokenizer.json")
        try:
            with open(rec_char_dict_path, "w") as f:
                json.dump(character_list, f)
        except Exception as e:
            print(f"创建 latexocr_tokenizer.json 文件失败, 原因{str(e)}")
        self.tokenizer = TokenizerFast.from_file(rec_char_dict_path)

    def post_process(self, s):
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s

    def decode(self, tokens):
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]

        dec = [self.tokenizer.decode(tok) for tok in tokens]
        dec_str_list = [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]
        return [str(self.post_process(dec_str)) for dec_str in dec_str_list]

    def apply(self, pred):
        key = next(iter(pred))
        preds = np.array(pred[key])
        text = self.decode(preds)
        return {"rec_text": text[0]}
