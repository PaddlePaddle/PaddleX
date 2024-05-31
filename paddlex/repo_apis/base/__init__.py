# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""

from .config import Config, BaseConfig
from .model import PaddleModel, BaseModel
from .runner import BaseRunner, InferOnlyRunner

# Init cache
from ...utils.cache import create_cache_dir

create_cache_dir()
