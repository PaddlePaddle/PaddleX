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

__all__ = ['DEBUG', 'DRY_RUN', 'CHECK_OPTS', 'EAGER_INITIALIZATION']


def get_flag_from_env_var(name, default):
    """ get_flag_from_env_var """
    env_var = os.environ.get(name, None)
    if env_var in ('True', 'true', 'TRUE', '1'):
        return True
    elif env_var in ('False', 'false', 'FALSE', '0'):
        return False
    else:
        return default


DEBUG = get_flag_from_env_var('PADDLE_PDX_DEBUG', False)
DRY_RUN = get_flag_from_env_var('PADDLE_PDX_DRY_RUN', False)
CHECK_OPTS = get_flag_from_env_var('PADDLE_PDX_CHECK_OPTS', False)
EAGER_INITIALIZATION = get_flag_from_env_var('PADDLE_PDX_EAGER_INIT', True)
