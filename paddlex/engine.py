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

from .modules.base import build_dataset_checker, build_trainer, build_evaluater
from .utils.result_saver import try_except_decorator
from .utils import config
from .utils.errors import raise_unsupported_api_error


class Engine(object):
    """ Engine """

    def __init__(self):
        args = config.parse_args()
        self.config = config.get_config(
            args.config, overrides=args.override, show=False)
        self.mode = self.config.Global.mode
        self.output_dir = self.config.Global.output

    @try_except_decorator
    def run(self):
        """ the main function """
        if self.config.Global.mode == "check_dataset":
            check_dataset = build_dataset_checker(self.config)
            return check_dataset()
        elif self.config.Global.mode == "train":
            train = build_trainer(self.config)
            train()
        elif self.config.Global.mode == "evaluate":
            evaluate = build_evaluater(self.config)
            return evaluate()
        elif self.config.Global.mode == "export":
            raise_unsupported_api_error("export", self.__class__)
        else:
            raise_unsupported_api_error(f"{self.config.Global.mode}",
                                        self.__class__)
