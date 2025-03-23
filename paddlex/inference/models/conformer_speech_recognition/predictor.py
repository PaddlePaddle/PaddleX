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

from typing import Any, Union, Dict, List, Tuple, Iterator
import numpy as np
from pathlib import Path
import tempfile
import shutil
from importlib import import_module

from ....utils import logging
from ....utils.func_register import FuncRegister
from ...common.batch_sampler import ConformerSpeechBatchSampler
from ...common.reader.chunk_conformer_reader import ReadChunkConformer
from ..common import StaticInfer
from ..base import BasicPredictor
from ..base.predictor.base_predictor import PredictionWrap
from .processors import (
    LoadAudioFromFile,
    ExtractFeatures,
    ProcessChunks,
    DecodeOutputs,
    GetInferInput,
)
from .result import ConformerSpeechResult

module_speech_recognition = import_module(
    ".conformer_speech_recognition", "paddlex.modules"
)
module_model_list = getattr(module_speech_recognition, "model_list")
MODELS = getattr(module_model_list, "MODELS")


class ConformerSpeechPredictor(BasicPredictor):
    """ConformerSpeechPredictor that inherits from BasicPredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes ConformerSpeechPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        self.temp_dir = tempfile.mkdtemp()
        logging.info(
            f"infer data will be stored in temporary directory {self.temp_dir}"
        )
        super().__init__(*args, **kwargs)
        # Set audio processing parameters before building the model
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.chunk_size = self.config.get("chunk_size", 16)  # in seconds
        self.stride = self.config.get("stride", 4)  # in seconds
        # Build the model after setting the parameters
        self.pre_tfs, self.infer = self._build()

    def _build_batch_sampler(self) -> ConformerSpeechBatchSampler:
        """Builds and returns a ConformerSpeechBatchSampler instance.

        Returns:
            ConformerSpeechBatchSampler: An instance of ConformerSpeechBatchSampler.
        """
        return ConformerSpeechBatchSampler(temp_dir=self.temp_dir)

    def _get_result_class(self) -> type:
        """Returns the result class, ConformerSpeechResult.

        Returns:
            type: The ConformerSpeechResult class.
        """
        return ConformerSpeechResult

    def _build(self) -> Tuple:
        """Build the preprocessors and inference engine based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors and inference engine.
        """
        # Convert seconds to samples for ReadChunkConformer
        chunk_size_samples = int(self.sample_rate * self.chunk_size)
        stride_samples = int(self.sample_rate * self.stride)
        pre_tfs = {
            "Read": ReadChunkConformer(
                chunk_size=chunk_size_samples, stride=stride_samples
            )
        }

        # Process the transform operations from config
        for cfg in self.config["PreProcess"]["transform_ops"]:
            tf_key = list(cfg.keys())[0]
            func = self._FUNC_MAP[tf_key]
            args = cfg.get(tf_key, {})
            name, op = func(self, **args) if args else func(self)
            if op:
                pre_tfs[name] = op
        pre_tfs["GetInferInput"] = GetInferInput()

        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        return pre_tfs, infer

    def _format_output(
        self, infer_input: List[Any], outs: List[Any], audio_metas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """format inference input and output into predict result

        Args:
            infer_input(List): Model infer inputs with list containing audio features.
            outs(List): Model infer output containing logits and decoded text.
            audio_metas(Dict): Audio metas info of input sample.

        Returns:
            Dict: A Dict containing formatted inference output results.
        """
        input_audio_path = audio_metas["input_audio_path"]
        sample_id = audio_metas["sample_id"]
        results = {}

        results["input_path"] = [input_audio_path]
        results["sample_id"] = [sample_id]
        results["logits"] = [outs[0]]
        results["text"] = [outs[1]]
        results["features"] = [infer_input[0]]

        return results

    def process(self, batch_data: List[str]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing and inference.

        Args:
            batch_data (List[str]): A batch of input data (e.g., audio file paths).

        Returns:
            dict: A dictionary containing the input path, features, output logits and decoded text.
        """
        sample = self.pre_tfs["Read"](batch_data=batch_data)
        sample = self.pre_tfs["LoadAudioFromFile"](results=sample[0])
        sample = self.pre_tfs["ExtractFeatures"](results=sample)
        sample = self.pre_tfs["ProcessChunks"](results=sample)
        infer_input, audio_metas = self.pre_tfs["GetInferInput"](sample=sample)
        infer_output = self.infer(x=infer_input)
        results = self._format_output(infer_input, infer_output, audio_metas)
        return results

    @register("LoadAudioFromFile")
    def build_load_audio_from_file(self, sample_rate=16000):
        return "LoadAudioFromFile", LoadAudioFromFile(sample_rate=sample_rate)

    @register("ExtractFeatures")
    def build_extract_features(
        self, n_fft=400, hop_length=160, win_length=None, window=None
    ):
        # If win_length is None, use n_fft as default
        if win_length is None:
            win_length = n_fft

        return "ExtractFeatures", ExtractFeatures(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window
        )

    @register("ProcessChunks")
    def build_process_chunks(self, chunk_size=16, stride=4):
        # Convert seconds to frames using hop_length and sample_rate
        # Assuming 10ms per frame (hop_length=160, sample_rate=16000)
        # So frames = seconds * sample_rate / hop_length
        chunk_size_frames = int(chunk_size * self.sample_rate / 160)
        context_frames = int(stride * self.sample_rate / 160)

        return "ProcessChunks", ProcessChunks(
            chunk_size_frames=chunk_size_frames,
            context_frames=context_frames,
        )

    @register("DecodeOutputs")
    def build_decode_outputs(self, vocab_path=None, decoding_method="ctc_greedy"):
        if vocab_path is None:
            vocab_path = Path(self.model_dir) / "vocab.txt"
        return "DecodeOutputs", DecodeOutputs(
            vocab_path=vocab_path,
            decoding_method=decoding_method,
        )

    @register("GetInferInput")
    def build_get_infer_input(self):
        return "GetInferInput", GetInferInput()

    def apply(self, input: Any, **kwargs) -> Iterator[Any]:
        """
        Do predicting with the input data and yields predictions.

        Args:
            input (Any): The input data to be predicted.

        Yields:
            Iterator[Any]: An iterator yielding prediction results.
        """
        try:
            for batch_data in self.batch_sampler(input):
                prediction = self.process(batch_data, **kwargs)
                prediction = PredictionWrap(prediction, len(batch_data))
                for idx in range(len(batch_data)):
                    yield self.result_class(prediction.get_by_idx(idx))
        except Exception as e:
            raise e
        finally:
            shutil.rmtree(self.temp_dir)
