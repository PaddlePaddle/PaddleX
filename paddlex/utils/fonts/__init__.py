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



from pathlib import Path


def get_pingfang_file_path():
    """ get pingfang font file path """
    return (Path(__file__).parent /
            "PingFang-SC-Regular.ttf").resolve().as_posix()


PINGFANG_FONT_FILE_PATH = get_pingfang_file_path()
