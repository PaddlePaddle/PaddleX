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

from ...utils.io import AudioReader


class ReadAudio:
    """Load audio from the file."""

    def __init__(self):
        """
        Initialize the instance.

        """
        super().__init__()
        self._audio_reader = AudioReader(backend="wav")

    def read(self, input):
        import paddle

        if isinstance(input, str):
            audio, sample_rate = self._audio_reader.read(input)
            if sample_rate != 16000:
                raise ValueError(
                    f"ReadAudio only supports 16k pcm or wav file.\n"
                    f"However, got: {sample_rate}."
                )
            audio = audio[:, 0]
            audio = paddle.to_tensor(audio)
            return audio, sample_rate
        else:
            raise TypeError(
                f"ReadAudio only supports str, indicating an audio file path.\n"
                f"However, got type: {type(input).__name__}."
            )
