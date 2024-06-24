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

import os

from . import version
from .modules import build_dataset_checker, build_trainer, build_evaluater, build_predictor
from .modules import create_model, PaddleInferenceOption
from .pipelines import *


def _initialize():
    from .utils.logging import setup_logging
    from .utils import flags
    from . import repo_manager
    from . import repo_apis

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, 'repo_manager', 'repos'), repo_apis)

    setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


_initialize()

__version__ = version.get_pdx_version()
