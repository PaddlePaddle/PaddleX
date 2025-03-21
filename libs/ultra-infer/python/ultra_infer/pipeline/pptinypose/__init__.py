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

from ... import c_lib_wrap as C


class PPTinyPose(object):
    def __init__(self, det_model=None, pptinypose_model=None):
        """Set initialized detection model object and pptinypose model object

        :param det_model: (ultra_infer.vision.detection.PicoDet)Initialized detection model object
        :param pptinypose_model: (ultra_infer.vision.keypointdetection.PPTinyPose)Initialized pptinypose model object
        """
        assert (
            det_model is not None or pptinypose_model is not None
        ), "The det_model and pptinypose_model cannot be None."
        self._pipeline = C.pipeline.PPTinyPose(
            det_model._model, pptinypose_model._model
        )

    def predict(self, input_image):
        """Predict the keypoint detection result for an input image

        :param im: (numpy.ndarray)The input image data, 3-D array with layout HWC, BGR format
        :return: KeyPointDetectionResult
        """
        return self._pipeline.predict(input_image)

    @property
    def detection_model_score_threshold(self):
        """Atrribute of PPTinyPose pipeline model. Stating the score threshold for detectin model to filter bbox before inputting pptinypose model

        :return: value of detection_model_score_threshold(float)
        """
        return self._pipeline.detection_model_score_threshold

    @detection_model_score_threshold.setter
    def detection_model_score_threshold(self, value):
        """Set attribute detection_model_score_threshold of PPTinyPose pipeline model.

        :param value: (float)The value to set use_dark
        """
        assert isinstance(
            value, float
        ), "The value to set `detection_model_score_threshold` must be type of float."
        self._pipeline.detection_model_score_threshold = value
