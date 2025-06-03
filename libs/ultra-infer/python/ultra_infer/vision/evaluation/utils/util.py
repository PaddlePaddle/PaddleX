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

import platform
import multiprocessing as mp


def is_pic(img_name):
    valid_suffix = ["JPEG", "jpeg", "JPG", "jpg", "BMP", "bmp", "PNG", "png"]
    suffix = img_name.split(".")[-1]
    if suffix not in valid_suffix:
        return False
    return True


def get_num_workers(num_workers):
    if not platform.system() == "Linux":
        # Dataloader with multi-process model is not supported
        # on MacOS and Windows currently.
        return 0
    if num_workers == "auto":
        num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 2 else 2
    return num_workers
