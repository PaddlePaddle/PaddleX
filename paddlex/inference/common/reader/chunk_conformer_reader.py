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

import numpy as np
import lazy_paddle as paddle
from ...utils.io import AudioReader
from ...utils.benchmark import benchmark


@benchmark.timeit_with_options(name=None, is_read_operation=True)
class ReadChunkConformer:
    """Load and process audio chunks for conformer models."""

    def __init__(self, chunk_size: int = 16000, stride: int = 4000):
        """
        Initialize chunk audio reader.

        Args:
            chunk_size (int): Size of audio chunks in samples. Default: 16000 (1 sec @16kHz)
            stride (int): Stride between chunks in samples. Default: 4000 (0.25 sec)
        """
        super().__init__()
        self.chunk_size = chunk_size
        self.stride = stride
        self._audio_reader = AudioReader(backend="wav")

    def read(self, input):
        if isinstance(input, str):
            # Load full audio and split into chunks
            audio, sample_rate = self._load_and_validate_audio(input)
            return self._chunk_audio(audio, sample_rate)
        elif isinstance(input, np.ndarray):
            # Process numpy array input
            audio = paddle.to_tensor(input)
            return self._chunk_audio(audio, sample_rate=16000)
        else:
            raise TypeError(
                f"ReadChunkConformer supports str paths or numpy arrays, but got {type(input)}"
            )

    def _load_and_validate_audio(self, path: str):
        """Load audio and validate sample rate."""
        audio, sample_rate = self._audio_reader.read(path)
        if sample_rate != 16000:
            raise ValueError(
                f"ChunkConformer requires 16kHz audio, got {sample_rate}Hz"
            )
        audio = audio[:, 0]  # Use mono channel
        return paddle.to_tensor(audio), sample_rate

    def _chunk_audio(self, audio: paddle.Tensor, sample_rate: int):
        """Split audio into chunks with overlap."""
        num_samples = audio.shape[0]
        chunks = []

        for start in range(0, num_samples, self.stride):
            end = start + self.chunk_size
            chunk = audio[start:end]

            # Pad last chunk if needed
            if chunk.shape[0] < self.chunk_size:
                pad_size = self.chunk_size - chunk.shape[0]
                chunk = paddle.concat([chunk, paddle.zeros(pad_size)], axis=0)

            chunks.append(chunk)

            # Stop if we've reached the end
            if end >= num_samples:
                break

        return chunks, sample_rate
