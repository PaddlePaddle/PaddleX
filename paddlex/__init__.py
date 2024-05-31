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

from .utils import flags
from . import version


def _initialize():
    from . import repo_manager
    from . import repo_apis

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, 'repo_manager', 'repos'), repo_apis)

    # setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


_initialize()

__version__ = version.get_pdx_version()
