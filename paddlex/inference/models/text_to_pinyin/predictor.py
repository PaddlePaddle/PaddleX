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

from ....modules.text_to_pinyin.model_list import MODELS
from ...common.batch_sampler import TextBatchSampler
from ..base import BasePredictor
from .result import TextToPinyinResult


class TextToPinyinPredictor(BasePredictor):

    entities = MODELS

    def __init__(self, *args, **kwargs):
        """Initializes TextSegmentPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.model = self._build()

    def _build_batch_sampler(self):
        """Builds and returns an TextBatchSampler instance.

        Returns:
            TextBatchSampler: An instance of TextBatchSampler.
        """
        return TextBatchSampler()

    def _get_result_class(self):
        """Returns the result class, TextToPinyinResult.

        Returns:
            type: The TextToPinyinResult class.
        """
        return TextToPinyinResult

    def _build(self):
        """Build the model.

        Returns:
            G2PWOnnxConverter: An instance of G2PWOnnxConverter.
        """
        from .processors import G2PWOnnxConverter

        # build model
        model = G2PWOnnxConverter(
            model_dir=self.model_dir, style="pinyin", enable_non_tradional_chinese=True
        )
        return model

    def process(self, batch_data):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str], ...]): A batch of input text data.

        Returns:
            dict: A dictionary containing the input path and result. The result include the output pinyin dict.
        """
        result = self.model(batch_data[0])
        return {"result": [result]}
