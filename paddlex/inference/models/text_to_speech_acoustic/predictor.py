# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

# import lazy_paddle as paddle
import paddle
import numpy as np

from ....utils.func_register import FuncRegister
from ...common.batch_sampler import AudioBatchSampler

from ..base import BasePredictor
from .result import Fastspeech2Result
from ....modules.text_to_speech_acoustic.model_list import MODELS


class Fastspeech2Predictor(BasePredictor):

    entities = MODELS

    def __init__(self, *args, **kwargs):
        """Initializes FastspeechPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.model = self._build()
    def _build_batch_sampler(self):
        """Builds and returns an AudioBatchSampler instance.

        Returns:
            AudioBatchSampler: An instance of AudioBatchSampler.
        """
        return AudioBatchSampler()

    def _get_result_class(self):
        """Returns the result class, Fastspeech2Result.

        Returns:
            type: The Fastspeech2Result class.
        """
        return Fastspeech2Result

    def _build(self):
        """Build the model.

        Returns:
            Fastspeech2: An instance of Fastspeech2.
        """
        from .processors import get_predictor
        model = get_predictor(
            model_dir=str(self.model_dir),
            model_file=self.config['Global']['model'] + ".pdmodel",
            params_file=self.config['Global']['model'] + ".pdiparams",
            device=self.config['Global']['device'],
            use_trt=self.config['Global']['use_trt'],
            use_mkldnn=self.config['Global']['use_mkldnn'],
            cpu_threads=self.config['Global']['cpu_threads'],
            precision=self.config['Global']['precision'],)
        return model

    def process(self, batch_data):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str], ...]): A batch of input phone data.

        Returns:
            dict: A dictionary containing the input path and result. The result include the output pinyin dict.
        """
        print(batch_data)
        from .processors import get_am_output
        phone = batch_data
        mel = get_am_output(
                input=phone,
                am_predictor=self.model,
                am=self.config['Global']['model'],
                lang=self.config['Global']['lang'],
                speaker_dict=self.config['Global']['speaker_dict'],
                spk_id=self.config['Global']['speaker_id'], 
        )
        result = mel
        print(result)
        return {
            "result": mel,
        }

