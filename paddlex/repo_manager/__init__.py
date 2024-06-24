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

from .core import (set_parent_dirs, setup, wheel, is_initialized, initialize,
                   get_versions)
from .meta import get_all_repo_names as get_all_supported_repo_names
