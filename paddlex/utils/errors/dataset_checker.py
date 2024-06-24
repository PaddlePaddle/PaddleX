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
__all__ = [
    'FailedError', 'CheckFailedError', 'ConvertFailedError', 'SplitFailedError',
    'AnalyseFailedError', 'CheckFailedError', 'DatasetFileNotFoundError'
]


class FailedError(Exception):
    """ base error class """

    def __init__(self, err_info=None, solution=None, message=None):
        if message is None:
            message = self._construct_message(err_info, solution)
        super().__init__(message)

    def _construct_message(self, err_info, solution):
        if err_info is None:
            return ""
        else:
            msg = f"{self.mode} failed. We encountered the following error:\n  {err_info}"
            if solution is not None:
                msg += f"\nPlease try to resolve the issue as follows:\n  {solution}"
            return msg


class CheckFailedError(FailedError):
    """ check dataset error """
    mode = "Check dataset"


class ConvertFailedError(FailedError):
    """ convert dataset error """
    mode = "Convert dataset"


class SplitFailedError(FailedError):
    """ split dataset error """
    mode = "Split dataset"


class AnalyseFailedError(FailedError):
    """ analyse dataset error """
    mode = "Analyse dataset"


class DatasetFileNotFoundError(CheckFailedError):
    """ dataset file not found error """

    def __init__(self,
                 file_path=None,
                 err_info=None,
                 solution=None,
                 message=None):
        if err_info is None:
            if file_path is not None:
                err_info = f"{file_path} does not exist."
        super().__init__(err_info, solution, message)
