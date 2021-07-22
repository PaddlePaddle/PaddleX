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

import sys
import glob
import os
import os.path as osp
import platform
import random
import numpy as np
import multiprocessing as mp
import paddle

from . import logging


def get_environ_info():
    """collect environment information"""

    env_info = dict()
    # TODO is_compiled_with_cuda() has not been moved
    compiled_with_cuda = paddle.is_compiled_with_cuda()
    if compiled_with_cuda:
        if 'gpu' in paddle.get_device():
            gpu_nums = paddle.distributed.get_world_size()
        else:
            gpu_nums = 0
        if gpu_nums == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
    place = 'gpu' if compiled_with_cuda and gpu_nums else 'cpu'
    env_info['place'] = place
    env_info['num'] = int(os.environ.get('CPU_NUM', 1))
    if place == 'gpu':
        env_info['num'] = gpu_nums

    return env_info


def get_num_workers(num_workers):
    if not platform.system() == 'Linux':
        # Dataloader with multi-process model is not supported
        # on MacOS and Windows currently.
        return 0
    if num_workers == 'auto':
        num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 2 else 2
    return num_workers


def init_parallel_env():
    env = os.environ
    if 'FLAGS_allocator_strategy' not in os.environ:
        os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
    dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    paddle.distributed.init_parallel_env()
