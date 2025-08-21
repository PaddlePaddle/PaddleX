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

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...models.text_to_speech_vocoder.result import PwganResult
from ...models.text_to_speech_acoustic.result import Fastspeech2Result
from ...models.text_to_pinyin.result import TextToPinyinResult
from ...utils.benchmark import benchmark
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline


@benchmark.time_methods
class TextToSpeechPipeline(BasePipeline):
    """Text to Speech Pipeline Pipeline"""

    entities = "text_to_speech"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        text_to_pinyin_model_config = config["SubModules"][
            "text_to_pinyin"
        ]
        self.text_to_pinyin_model = self.create_model(
            text_to_pinyin_model_config
        )
        text_to_speech_acoustic_model_config = config["SubModules"][
            "text_to_speech_acoustic"
        ]
        self.text_to_speech_acoustic_model = self.create_model(
            text_to_speech_acoustic_model_config
        )
        text_to_speech_vocoder_model_config = config["SubModules"][
            "text_to_speech_vocoder"
        ]
        self.text_to_speech_vocoder_model = self.create_model(
            text_to_speech_vocoder_model_config
        )

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> WhisperResult:
        """Predicts speech recognition results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input audio or path.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            WhisperResult: The predicted whisper results, support str and json output.
        """
        if type(input) == str:
            if input.endswith("txt"):
                if os.path.exists(input):
                    with open(input, "r", encoding="utf-8") as f:
                        sentences = f.readlines()
            else:
                sentences = [input]
        elif type(input) == list:
            sentences = input
        else:
            raise TypeError("Input must be string or list of strings.")
        for sentence in sentences:
            text_to_pinyin_res = get_text_to_pinyin_result(sentence)
            text_to_speech_acoustic_res = get_text_to_speech_acoustic_result(text_to_pinyin_res)
            yield from self.text_to_speech_vocoder_model(text_to_speech_acoustic_res)

    def get_text_to_pinyin_result(self, input: Union[str, List[str]]
    ) -> TextToPinyinResult:
        """Get the result of text to pinyin conversion.

        Args:
            input (Union[str, list[str]]): The input text or list of texts.

        Returns:
            TextToPinyinResult: The result of text to pinyin conversion.
        """
        return next(self.text_to_pinyin_model(input))
    
    def get_text_to_speech_acoustic_result(self, input: Union[str, List[str]]
    ) -> Fastspeech2Result:
        """Get the result of text to speech acoustic conversion.

        Args:
            input (Union[str, list[str]]): The input text or list of texts.

        Returns:
            Fastspeech2Result: The result of text to speech acoustic conversion.
        """
        return next(self.text_to_speech_acoustic_model(input))

