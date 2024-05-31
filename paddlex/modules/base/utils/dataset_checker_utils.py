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

# TODO: this file is need to be refactor

import os

from .cache import persist

__all__ = [
    'RES_DICT_TYPE', 'build_res_dict', 'persist_dataset_meta',
    'assert_no_backslash', 'assert_valid_path', 'CheckFailedError',
    'UnsupportedDatasetTypeError', 'DatasetFileNotFoundError',
    'BackslashInPathError'
]

RES_DICT_TYPE = dict


def build_res_dict(res_flag, err_type=None, err_msg=None, **kwargs):
    """ build res dict """
    if res_flag:
        if err_type is not None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_type` is not None.")
        if err_msg is not None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_msg` is not None.")
        return RES_DICT_TYPE(res_flag=res_flag, **kwargs)
    else:
        if err_type is None:
            raise ValueError(
                f"`res_flag` is {res_flag}, but `err_type` is None.")
        if err_msg is None:
            if _is_known_error_type(err_type):
                err_msg = ""
            else:
                raise ValueError(
                    f"{err_type} is not a known error type, in which case `err_msg` must be specified to a value \
other than None.")
        return RES_DICT_TYPE(
            res_flag=res_flag, err_type=err_type, err_msg=err_msg, **kwargs)


def assert_no_backslash(path):
    """ assert no backslash """
    if '\\' in path:
        raise BackslashInPathError(path=path)


def assert_valid_path(path):
    """ assert valid path """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise DatasetFileNotFoundError(file_path=path)


class CheckFailedError(Exception):
    """ CheckFailedError """

    def __init__(self, err_info=None, solution=None, message=None):
        if message is None:
            message = self._construct_message(err_info, solution)
        super().__init__(message)

    def _construct_message(self, err_info, solution):
        """ construct message """
        if err_info is None:
            return ""
        else:
            msg = f"Dataset check failed. We encountered the following error:\n  {err_info}"
            if solution is not None:
                msg += f"\nPlease try to resolve the issue as follows:\n  {solution}"
            return msg


class UnsupportedDatasetTypeError(CheckFailedError):
    """ UnsupportedDatasetTypeError """

    def __init__(self,
                 dataset_type=None,
                 err_info=None,
                 solution=None,
                 message=None):
        if err_info is None:
            if dataset_type is not None:
                err_info = f"{repr(dataset_type)} is not a supported dataset type."
        super().__init__(err_info, solution, message)


class DatasetFileNotFoundError(CheckFailedError):
    """ DatasetFileNotFoundError """

    def __init__(self,
                 file_path=None,
                 err_info=None,
                 solution=None,
                 message=None):
        if err_info is None:
            if file_path is not None:
                err_info = f"{file_path} does not exist."
        super().__init__(err_info, solution, message)


class BackslashInPathError(CheckFailedError):
    """ BackslashInPathError """

    def __init__(self, path=None, err_info=None, solution=None, message=None):
        if err_info is None:
            if path is not None:
                err_info = (
                    f"{path} contains backslashes, which is not supported. "
                    "If a backslash ('\\') is used as the path separator, please replace \
it with a forward slash ('/'). "
                    "For example, please change 'dir\\file' to 'dir/file'.")
        super().__init__(err_info, solution, message)


def _is_valid_dataset(res_dict):
    """ is valid dataset """
    assert isinstance(res_dict, RES_DICT_TYPE)
    flag = res_dict['res_flag']
    return flag


def _is_known_error_type(err_type):
    """ is known error type """
    return isinstance(err_type, CheckFailedError)


persist_dataset_meta = persist(cond=_is_valid_dataset)
