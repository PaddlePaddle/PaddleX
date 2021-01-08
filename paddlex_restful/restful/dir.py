#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp


def gen_user_home():
    if "PADDLE_HOME" in os.environ:
        home_path = os.environ["PADDLE_HOME"]
        if os.path.exists(home_path) and os.path.isdir(home_path):
            return home_path
    return os.path.expanduser('~')


def gen_paddlex_home():
    path = osp.join(gen_user_home(), ".paddlex_server")
    if not osp.exists(path):
        os.makedirs(path)
    return path


USER_HOME = gen_user_home()
PADDLEX_HOME = gen_paddlex_home()
WORKSPACE_HOME = osp.join(USER_HOME, "paddlex_workspace")
LOG_HOME = osp.join(PADDLEX_HOME, "logs")
SINGLE_LOCK_HOME = osp.join(PADDLEX_HOME, "single_lock")
CACHE_HOME = osp.join(PADDLEX_HOME, "cache")

for home in [WORKSPACE_HOME, LOG_HOME, SINGLE_LOCK_HOME, CACHE_HOME]:
    if not osp.exists(home):
        os.makedirs(home)
