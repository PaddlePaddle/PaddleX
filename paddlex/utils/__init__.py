# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import logging
from . import utils
from .utils import (seconds_to_hms, get_encoding, get_single_card_bs, dict2str,
                    EarlyStop, path_normalization, is_pic, MyEncoder,
                    DisablePrint)
from .checkpoint import get_pretrain_weights, load_pretrain_weights, load_checkpoint
from .env import get_environ_info, get_num_workers, init_parallel_env
from .download import download_and_decompress, decompress
from .stats import SmoothedValue, TrainingStats
from .shm import _get_shared_memory_size_in_M
