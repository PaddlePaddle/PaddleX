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

import os
import shutil
import sys


def copy_directory(src, dst):
    if os.path.exists(dst):
        raise Exception("Destination {} is already exist.".format(dst))
    if not os.path.exists(src):
        raise Exception("Source {} is not exist.".format(src))
    try:
        shutil.copytree(src, dst, symlinks=True)
    except:
        raise Exception("Copy {} to {} failed.".format(src, dst))


if __name__ == "__main__":
    copy_directory(sys.argv[1], sys.argv[2])
