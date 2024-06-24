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


class FromDictMixin(object):
    """ FromDictMixin """

    @classmethod
    def from_dict(cls, dict_):
        """ from dict """
        return cls(**dict_)
