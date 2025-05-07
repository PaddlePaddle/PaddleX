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
from dataclasses import dataclass
from typing import List

import numpy as np

from .... import UltraInferModel, ModelFormat
from .... import c_lib_wrap as C
from ...common import ProcessorManager
from ....py_only import PyOnlyProcessorChain
from ....py_only.vision import PyOnlyVisionModel, processors as P
from ....utils.misc import load_config


class PaddleSegModel(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a image segmentation model exported by PaddleSeg.

        :param model_file: (str)Path of model file, e.g unet/model.pdmodel
        :param params_file: (str)Path of parameters file, e.g unet/model.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str) Path of configuration file for deploy, e.g unet/deploy.yml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """
        super(PaddleSegModel, self).__init__(runtime_option)

        # assert model_format == ModelFormat.PADDLE, "PaddleSeg only support model format of ModelFormat.Paddle now."
        self._model = C.vision.segmentation.PaddleSegModel(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleSeg model initialize failed."

    def predict(self, image):
        """Predict the segmentation result for an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: SegmentationResult
        """
        return self._model.predict(image)

    def batch_predict(self, image_list):
        """Predict the segmentation results for a batch of input images

        :param image_list: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return: list of SegmentationResult
        """
        return self._model.batch_predict(image_list)

    def clone(self):
        """Clone PaddleSegModel object

        :return: a new PaddleSegModel object
        """

        class PaddleSegCloneModel(PaddleSegModel):
            def __init__(self, model):
                self._model = model

        clone_model = PaddleSegCloneModel(self._model.clone())
        return clone_model

    @property
    def preprocessor(self):
        """Get PaddleSegPreprocessor object of the loaded model

        :return: PaddleSegPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleSegPostprocessor object of the loaded model

        :return: PaddleSegPostprocessor
        """
        return self._model.postprocessor


class PaddleSegPreprocessor(ProcessorManager):
    def __init__(self, config_file):
        """Create a preprocessor for PaddleSegModel from configuration file

        :param config_file: (str)Path of configuration file, e.g ppliteseg/deploy.yaml
        """
        self._manager = C.vision.segmentation.PaddleSegPreprocessor(config_file)

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

    @property
    def is_vertical_screen(self):
        """Atrribute of PP-HumanSeg model. Stating Whether the input image is vertical image(height > width), default value is False

        :return: value of is_vertical_screen(bool)
        """
        return self._manager.is_vertical_screen

    @is_vertical_screen.setter
    def is_vertical_screen(self, value):
        """Set attribute is_vertical_screen of PP-HumanSeg model.

        :param value: (bool)The value to set is_vertical_screen
        """
        assert isinstance(
            value, bool
        ), "The value to set `is_vertical_screen` must be type of bool."
        self._manager.is_vertical_screen = value


class PaddleSegPostprocessor:
    def __init__(self, config_file):
        """Create a postprocessor for PaddleSegModel from configuration file

        :param config_file: (str)Path of configuration file, e.g ppliteseg/deploy.yaml
        """
        self._postprocessor = C.vision.segmentation.PaddleSegPostprocessor(config_file)

    def run(self, runtime_results, imgs_info):
        """Postprocess the runtime results for PaddleSegModel

        :param runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :param imgs_info: The original input images shape info map, key is "shape_info", value is [[image_height, image_width]]
        :return: list of SegmentationResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results, imgs_info)

    @property
    def apply_softmax(self):
        """Atrribute of PaddleSeg model. Stating Whether applying softmax operator in the postprocess, default value is False

        :return: value of apply_softmax(bool)
        """
        return self._postprocessor.apply_softmax

    @apply_softmax.setter
    def apply_softmax(self, value):
        """Set attribute apply_softmax of PaddleSeg model.

        :param value: (bool)The value to set apply_softmax
        """
        assert isinstance(
            value, bool
        ), "The value to set `apply_softmax` must be type of bool."
        self._postprocessor.apply_softmax = value

    @property
    def store_score_map(self):
        """Atrribute of PaddleSeg model. Stating Whether storing score map in the SegmentationResult, default value is False

        :return: value of store_score_map(bool)
        """
        return self._postprocessor.store_score_map

    @store_score_map.setter
    def store_score_map(self, value):
        """Set attribute store_score_map of PaddleSeg model.

        :param value: (bool)The value to set store_score_map
        """
        assert isinstance(
            value, bool
        ), "The value to set `store_score_map` must be type of bool."
        self._postprocessor.store_score_map = value


class PyOnlyAnomalyDetectionModel(PyOnlyVisionModel):
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
        self._preprocessor = _PyOnlyAnomalyDetectionPreprocessor(
            self._config["Deploy"]["transforms"]
        )
        self._postprocessor = _PyOnlyAnomalyDetectionPostprocessor()

    def model_name():
        return "PyOnlyImageAnomalyDetectionModel"

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


class _PyOnlyAnomalyDetectionPreprocessor(object):
    def __init__(self, config):
        super().__init__()
        processors = self._build_processors(config)
        processors.append(P.ToCHWImage())
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        return self._processor_chain(data)

    def _build_processors(self, config):
        processors = []
        for item in config:
            tf_type = item["type"]
            args = {k: v for k, v in item.items() if k != "type"}
            if tf_type == "Resize":
                if args.keys() - {
                    "target_size",
                    "keep_ratio",
                    "size_divisor",
                    "interp",
                }:
                    raise ValueError
                args.setdefault("keep_ratio", False)
                args.setdefault("size_divisor", None)
                args.setdefault("interp", "LINEAR")
                processor = P.Resize(
                    target_size=args["target_size"],
                    keep_ratio=args["keep_ratio"],
                    size_divisor=args["size_divisor"],
                    interp=args["interp"],
                )
            elif tf_type == "ResizeByLong":
                if args.keys() - {"long_size"}:
                    raise ValueError
                args.setdefault("size_divisor", None)
                args.setdefault("interp", "LINEAR")
                processor = P.ResizeByLong(target_long_edge=args["long_size"])
            elif tf_type == "ResizeByShort":
                if args.keys() - {"short_size"}:
                    raise ValueError
                processor = P.ResizeByShort(target_short_edge=args["short_size"])
            elif tf_type == "Normalize":
                if args.keys() - {"mean", "std"}:
                    raise ValueError
                args.setdefault("mean", 0.5)
                args.setdefault("std", 0.5)
                processor = P.Normalize(mean=args["mean"], std=args["std"])
            else:
                raise ValueError("Unknown transform type")
            processors.append(processor)
        return processors


class _PyOnlyAnomalyDetectionPostprocessor(object):
    def run(self, data):
        from skimage import morphology

        score_map = data["score_map"]

        thred = 0.01
        mask = score_map[0]
        mask[mask > thred] = 255
        mask[mask <= thred] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask = mask.astype(np.uint8)

        result = _PyOnlyAnomalyDetectionResult(
            label_map=mask.reshape((-1)).tolist(), shape=list(mask.shape)
        )
        return result


@dataclass
class _PyOnlyAnomalyDetectionResult(object):
    label_map: List[int]
    shape: List[int]
