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

from pathlib import Path


def get_pingfang_file_path():
    """ get pingfang font file path """
    return (Path(__file__).parent /
            "PingFang-SC-Regular.ttf").resolve().as_posix()


PINGFANG_FONT_FILE_PATH = get_pingfang_file_path()
