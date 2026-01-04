# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

from ....modules.text_to_speech_vocoder.model_list import MODELS
from ...common.batch_sampler import AudioBatchSampler
from ..base import BasePredictor
from .result import PwganResult


class PwganPredictor(BasePredictor):

    entities = MODELS

    def __init__(self, *args, **kwargs):
        """Initializes FastspeechPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.infer = self.create_static_infer()

    def _build_batch_sampler(self):
        """Builds and returns an AudioBatchSampler instance.

        Returns:
            AudioBatchSampler: An instance of AudioBatchSampler.
        """
        return AudioBatchSampler()

    def _get_result_class(self):
        """Returns the result class, PwganResult.

        Returns:
            type: The PwganResult class.
        """
        return PwganResult

    def process(self, batch_data):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str], ...]): A batch of input phone data.

        Returns:
            dict: A dictionary containing the input path and result. The result include the output pinyin dict.
        """
        input_data = batch_data[0]
        if type(input_data) is str:
            mel = np.load(input_data)
        else:
            mel = input_data
        wav = self.infer([mel])
        result = np.array(wav).reshape(1, -1)
        return {
            "result": result,
        }
