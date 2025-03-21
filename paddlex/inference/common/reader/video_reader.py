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


from ...utils.benchmark import benchmark
from ...utils.io import VideoReader


@benchmark.timeit_with_options(name=None, is_read_operation=True)
class ReadVideo:
    """Load video from the file."""

    def __init__(self, backend="opencv", num_seg=8, seg_len=1, sample_type=None):

        super().__init__()
        self._video_reader = VideoReader(
            backend=backend, num_seg=num_seg, seg_len=seg_len, sample_type=sample_type
        )

    def __call__(self, videos):
        """apply"""
        return [self._read(video) for video in videos]

    def _read(self, file_path):
        return self._read_video(file_path)

    def _read_video(self, video_path):
        blob = list(self._video_reader.read(video_path))
        if blob is None:
            raise Exception("Video read Error")
        return blob
