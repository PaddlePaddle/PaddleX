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

import abc

from .utils.mixin import FromDictMixin
from .utils.batch import batchable_method
from .utils.node import Node


class BaseTransform(FromDictMixin, Node):
    """ BaseTransform """

    @batchable_method
    def __call__(self, data):
        self.check_input_keys(data)
        data = self.apply(data)
        self.check_output_keys(data)
        return data

    @abc.abstractmethod
    def apply(self, data):
        """ apply """
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__
