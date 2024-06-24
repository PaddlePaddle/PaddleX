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
import platform
from pathlib import Path

import pandas as pd
import numpy as np


def deep_analyse(dataset_dir, output):
    """class analysis for dataset"""

    return {"histogram": ""}