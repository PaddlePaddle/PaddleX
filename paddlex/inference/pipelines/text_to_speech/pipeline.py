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

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ...models import HPIConfig, PaddlePredictorOption
from ...models.text_to_pinyin.result import TextToPinyinResult
from ...models.text_to_speech_acoustic.result import Fastspeech2Result
from ...models.text_to_speech_vocoder.result import PwganResult
from ...utils.benchmark import benchmark
from ..base import BasePipeline


@benchmark.time_methods
class TextToSpeechPipeline(BasePipeline):
    """Text to Speech Pipeline Pipeline"""

    entities = "text_to_speech"

    def __init__(
        self,
        config: Dict,
        *,
        device: Optional[str] = None,
        engine: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
        **kwargs,
    ) -> None:
        """Initializes the text-to-speech pipeline.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (Optional[str], optional): The device to use for prediction. Defaults to `None`.
            engine (Optional[str], optional): Inference engine. Defaults to `None`.
            engine_config (Optional[Dict[str, Any]], optional): Engine-specific config. Defaults to `None`.
            pp_option (Optional[PaddlePredictorOption], optional): Paddle predictor options.
                Defaults to `None`.
            use_hpip (bool, optional): Whether to use HPIP. Defaults to `False`.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                HPIP configuration. Defaults to `None`.
        """
        super().__init__(
            device=device,
            engine=engine,
            engine_config=engine_config,
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
            **kwargs,
        )

        text_to_pinyin_model_config = config["SubModules"]["TextToPinyin"]
        self.text_to_pinyin_model = self.create_model(text_to_pinyin_model_config)
        text_to_speech_acoustic_model_config = config["SubModules"][
            "TextToSpeechAcoustic"
        ]
        self.text_to_speech_acoustic_model = self.create_model(
            text_to_speech_acoustic_model_config
        )
        text_to_speech_vocoder_model_config = config["SubModules"][
            "TextToSpeechVocoder"
        ]
        self.text_to_speech_vocoder_model = self.create_model(
            text_to_speech_vocoder_model_config
        )

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> PwganResult:
        """Predicts speech recognition results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input audio or path.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            PwganResult: The predicted pwgan results, support str and json output.
        """
        sentences = []
        if isinstance(input, str):
            if input.endswith(".txt"):
                if not os.path.exists(input):
                    raise FileNotFoundError(
                        f"The specified text file does not exist: {input}"
                    )
                try:
                    with open(input, "r", encoding="utf-8") as f:
                        sentences = [line.strip() for line in f.readlines()]
                except IOError as e:
                    raise IOError(
                        f"An error occurred while reading the file {input}: {e}"
                    )
            else:
                sentences = [input]
        elif isinstance(input, list):
            for item in input:
                if isinstance(item, str):
                    if item.endswith(".txt"):
                        if not os.path.exists(item):
                            raise FileNotFoundError(
                                f"The specified text file in the list does not exist: {item}"
                            )
                        try:
                            with open(item, "r", encoding="utf-8") as f:
                                sentences.extend(
                                    [line.strip() for line in f.readlines()]
                                )
                        except IOError as e:
                            raise IOError(
                                f"An error occurred while reading the file {item}: {e}"
                            )
                    else:
                        sentences.append(item)
        else:
            raise TypeError(
                f"Unsupported input type: {type(input)}. Expected str, list, or np.ndarray."
            )
        if not sentences:
            raise ValueError(
                "The input resulted in an empty list of sentences to process."
            )

        for sentence in sentences:
            text_to_pinyin_res = [
                self.get_text_to_pinyin_result(sentence)["result"]["phone_ids"]
            ]
            text_to_speech_acoustic_res = [
                self.get_text_to_speech_acoustic_result(text_to_pinyin_res)["result"]
            ]
            yield from self.text_to_speech_vocoder_model(text_to_speech_acoustic_res)

    def get_text_to_pinyin_result(
        self, input: Union[str, List[str]]
    ) -> TextToPinyinResult:
        """Get the result of text to pinyin conversion.

        Args:
            input (Union[str, list[str]]): The input text or list of texts.

        Returns:
            TextToPinyinResult: The result of text to pinyin conversion.
        """
        return next(self.text_to_pinyin_model(input))

    def get_text_to_speech_acoustic_result(
        self, input: Union[str, List[str]]
    ) -> Fastspeech2Result:
        """Get the result of text to speech acoustic conversion.

        Args:
            input (Union[str, list[str]]): The input text or list of texts.

        Returns:
            Fastspeech2Result: The result of text to speech acoustic conversion.
        """
        return next(self.text_to_speech_acoustic_model(input))
