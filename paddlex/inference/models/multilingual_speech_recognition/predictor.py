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

from typing import Any, Dict, Tuple

import numpy as np

from .... import constants
from ....modules.multilingual_speech_recognition.model_list import MODELS
from ....utils.device import TemporaryDeviceChanger
from ....utils.download import download_and_extract
from ...common.batch_sampler import AudioBatchSampler
from ...utils.io import AudioReader
from ..predictors import RunnerPredictor
from ..runners import PaddleDynamicRunner
from .result import WhisperResult


class WhisperRunnerPredictor(RunnerPredictor):

    entities = MODELS

    @classmethod
    def get_supported_engines(cls) -> Tuple[str, ...]:
        return ("paddle_dynamic",)

    def __init__(self, *args, **kwargs):
        """Initializes WhisperPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        download_and_extract(self.config["resource_path"], self.model_dir, "assets")
        self.audio_reader = AudioReader(backend="wav")
        self.infer = self.create_runner()

    def _build_batch_sampler(self):
        """Builds and returns an AudioBatchSampler instance.

        Returns:
            AudioBatchSampler: An instance of AudioBatchSampler.
        """
        return AudioBatchSampler()

    def _get_result_class(self):
        """Returns the result class, WhisperResult.

        Returns:
            type: The WhisperResult class.
        """
        return WhisperResult

    def build_paddle_dynamic_runner(self) -> PaddleDynamicRunner:
        import paddle

        from .processors import ModelDimensions, Whisper

        with TemporaryDeviceChanger(self.device):
            model_file = (
                self.model_dir / f"{constants.MODEL_FILE_PREFIX}.pdparams"
            ).as_posix()
            model_dict = paddle.load(model_file)
            dims = ModelDimensions(**model_dict["dims"])
            model = Whisper(dims)
            model.load_dict(model_dict)
            model.eval()

        resource_path = self.model_dir.as_posix()

        def infer_fn(model: Any, inputs: Dict[str, Any]) -> np.ndarray:
            result = model.transcribe(
                inputs["mel"],
                resource_path=resource_path,
                **inputs["decode_kwargs"],
            )
            return np.array([result], dtype=object)

        return PaddleDynamicRunner(
            model,
            config=self._engine_config,
            infer_fn=infer_fn,
        )

    def _build_temperature(self):
        temperature_increment_on_fallback = self.config[
            "temperature_increment_on_fallback"
        ]
        if (
            temperature_increment_on_fallback is not None
            and temperature_increment_on_fallback != "None"
        ):
            return tuple(
                np.arange(
                    self.config["temperature"],
                    1.0 + 1e-6,
                    temperature_increment_on_fallback,
                )
            )
        return [self.config["temperature"]]

    def _build_decode_kwargs(self) -> Dict[str, Any]:
        return {
            "verbose": self.config["verbose"],
            "task": self.config["task"],
            "language": self.config["language"],
            "temperature": self._build_temperature(),
            "compression_ratio_threshold": self.config["compression_ratio_threshold"],
            "logprob_threshold": self.config["logprob_threshold"],
            "best_of": self.config["best_of"],
            "beam_size": self.config["beam_size"],
            "patience": self.config["patience"],
            "length_penalty": self.config["length_penalty"],
            "initial_prompt": self.config["initial_prompt"],
            "condition_on_previous_text": self.config["condition_on_previous_text"],
            "no_speech_threshold": self.config["no_speech_threshold"],
        }

    def _build_mel(self, input_data):
        import paddle

        from .processors import log_mel_spectrogram

        audio, _ = self.audio_reader.read(input_data)
        audio = paddle.to_tensor(audio)
        audio = audio[:, 0]
        return log_mel_spectrogram(audio, resource_path=self.model_dir)

    def process(self, batch_data):
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., audio file paths).

        Returns:
            dict: A dictionary containing the input path and result. The result include 'text', 'segments' and 'language'.
        """
        input_data = batch_data[0]
        mel = self._build_mel(input_data)
        result = self.infer(
            x={
                "mel": mel,
                "decode_kwargs": self._build_decode_kwargs(),
            }
        )
        result = result[0].item()
        return {
            "result": [result],
        }
