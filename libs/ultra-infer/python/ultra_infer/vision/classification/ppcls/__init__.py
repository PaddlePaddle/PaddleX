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

from .... import ModelFormat, UltraInferModel
from .... import c_lib_wrap as C
from ....py_only import PyOnlyProcessorChain
from ....py_only.vision import PyOnlyVisionModel
from ....py_only.vision import processors as P
from ....utils.misc import load_config
from ...common import ProcessorManager


class PaddleClasPreprocessor(ProcessorManager):
    def __init__(self, config_file):
        """Create a preprocessor for PaddleClasModel from configuration file

        :param config_file: (str)Path of configuration file, e.g resnet50/inference_cls.yaml
        """
        super(PaddleClasPreprocessor, self).__init__()
        self._manager = C.vision.classification.PaddleClasPreprocessor(config_file)

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


class PaddleClasPostprocessor:
    def __init__(self, topk=1):
        """Create a postprocessor for PaddleClasModel

        :param topk: (int)Filter the top k classify label
        """
        self._postprocessor = C.vision.classification.PaddleClasPostprocessor(topk)

    def run(self, runtime_results):
        """Postprocess the runtime results for PaddleClasModel

        :param: runtime_results: (list of FDTensor)The output FDTensor results from runtime
        :return: list of ClassifyResult(If the runtime_results is predict by batched samples, the length of this list equals to the batch size)
        """
        return self._postprocessor.run(runtime_results)


class PaddleClasModel(UltraInferModel):
    def __init__(
        self,
        model_file,
        params_file,
        config_file,
        runtime_option=None,
        model_format=ModelFormat.PADDLE,
    ):
        """Load a image classification model exported by PaddleClas.

        :param model_file: (str)Path of model file, e.g resnet50/inference.pdmodel
        :param params_file: (str)Path of parameters file, e.g resnet50/inference.pdiparams, if the model_fomat is ModelFormat.ONNX, this param will be ignored, can be set as empty string
        :param config_file: (str) Path of configuration file for deploy, e.g resnet50/inference_cls.yaml
        :param runtime_option: (ultra_infer.RuntimeOption)RuntimeOption for inference this model, if it's None, will use the default backend on CPU
        :param model_format: (ultra_infer.ModelForamt)Model format of the loaded model
        """

        super(PaddleClasModel, self).__init__(runtime_option)
        self._model = C.vision.classification.PaddleClasModel(
            model_file, params_file, config_file, self._runtime_option, model_format
        )
        assert self.initialized, "PaddleClas model initialize failed."

    def clone(self):
        """Clone PaddleClasModel object

        :return: a new PaddleClasModel object
        """

        class PaddleClasCloneModel(PaddleClasModel):
            def __init__(self, model):
                self._model = model

        clone_model = PaddleClasCloneModel(self._model.clone())
        return clone_model

    def predict(self, im, topk=1):
        """Classify an input image

        :param im: (numpy.ndarray) The input image data, a 3-D array with layout HWC, BGR format
        :param topk: (int) Filter the topk classify result, default 1
        :return: ClassifyResult
        """

        self.postprocessor.topk = topk
        return self._model.predict(im)

    def batch_predict(self, images):
        """Classify a batch of input image

        :param im: (list of numpy.ndarray) The input image list, each element is a 3-D array with layout HWC, BGR format
        :return list of ClassifyResult
        """

        return self._model.batch_predict(images)

    @property
    def preprocessor(self):
        """Get PaddleClasPreprocessor object of the loaded model

        :return PaddleClasPreprocessor
        """
        return self._model.preprocessor

    @property
    def postprocessor(self):
        """Get PaddleClasPostprocessor object of the loaded model

        :return PaddleClasPostprocessor
        """
        return self._model.postprocessor


class _PyOnlyMultilabelClassificationPreprocessor(object):
    def __init__(self, config):
        super().__init__()
        processors = self._build_processors(config)
        processors.insert(0, P.BGR2RGB())
        self._processor_chain = PyOnlyProcessorChain(processors)

    def run(self, data):
        return self._processor_chain(data)

    def _build_processors(self, config):
        processors = []
        for item in config:
            tf_type = next(iter(item))
            args = item[tf_type]
            if tf_type == "ResizeImage":
                if args.keys() - {"resize_short", "size", "backend", "interpolation"}:
                    raise ValueError
                args.setdefault("resize_short", None)
                args.setdefault("size", None)
                # TODO: `backend` & `interpolation`
                if not (args["resize_short"] or args["size"]):
                    raise ValueError
                if args.get("resize_short"):
                    processor = P.ResizeByShort(
                        target_short_edge=args["resize_short"],
                        size_divisor=None,
                        interp="LINEAR",
                    )
                else:
                    processor = P.Resize(target_size=args["size"])
            elif tf_type == "CropImage":
                if args.keys() - {"size"}:
                    raise ValueError
                args.setdefault("size", 224)
                processor = P.Crop(crop_size=args["size"])
            elif tf_type == "NormalizeImage":
                if args.keys() - {"mean", "std", "scale", "order", "channel_num"}:
                    raise ValueError
                args.setdefault("mean", [0.485, 0.456, 0.406])
                args.setdefault("std", [0.229, 0.224, 0.225])
                args.setdefault("scale", 1 / 255)
                args.setdefault("order", "")
                args.setdefault("channel_num", 3)
                if args["order"] != "":
                    raise ValueError
                if args["channel_num"] != 3:
                    raise ValueError
                processor = P.Normalize(
                    scale=args["scale"], mean=args["mean"], std=args["std"]
                )
            elif tf_type == "ToCHWImage":
                if args:
                    raise ValueError
                processor = P.ToCHWImage()
            else:
                raise ValueError("Unknown transform type")
            processors.append(processor)
        return processors


@dataclass
class _PyOnlyMultilabelClassificationResult(object):
    label_ids: List[int]
    scores: List[float]


class _PyOnlyMultilabelClassificationPostprocessor(object):
    def __init__(self, config):
        super().__init__()
        self._threshold = config["threshold"]

    def run(self, data):
        pred = data["pred"]

        pred_index = np.where(pred >= self._threshold)[0].astype("int32")
        index = pred_index[np.argsort(pred[pred_index])][::-1]
        clas_id_list = []
        score_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(pred[i].item())

        result = _PyOnlyMultilabelClassificationResult(
            label_ids=clas_id_list, scores=score_list
        )
        return result


class PyOnlyMultilabelClassificationModel(PyOnlyVisionModel):
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
        self._preprocessor = _PyOnlyMultilabelClassificationPreprocessor(
            self._config["PreProcess"]["transform_ops"]
        )
        self._postprocessor = _PyOnlyMultilabelClassificationPostprocessor(
            self._config["PostProcess"]["MultiLabelThreshOutput"]
        )

    def model_name():
        return "PyOnlyMultilabelImageClassificationModel"

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
        for pred in output_arrs[0]:
            data = {"pred": pred}
            result = self._postprocessor.run(data)
            results.append(result)
        return results

    def _update_option(self):
        self._option.set_model_path(
            self._model_file, self._params_file, self._model_format
        )
