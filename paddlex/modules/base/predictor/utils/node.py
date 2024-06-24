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

import inspect
import functools

from .....utils.misc import AutoRegisterABCMetaClass


class _KeyMissingError(Exception):
    """ _KeyMissingError """
    pass


class _NodeMeta(AutoRegisterABCMetaClass):
    """ _Node Meta Class """

    def __new__(cls, name, bases, attrs):
        def _deco(init_func):
            @functools.wraps(init_func)
            def _wrapper(self, *args, **kwargs):
                if not hasattr(self, '_raw_args'):
                    sig = inspect.signature(init_func)
                    bnd_args = sig.bind(self, *args, **kwargs)
                    raw_args = bnd_args.arguments
                    self_key = next(iter(raw_args.keys()))
                    raw_args.pop(self_key)
                    setattr(self, '_raw_args', raw_args)
                ret = init_func(self, *args, **kwargs)
                return ret

            return _wrapper

        if '__init__' in attrs:
            old_init_func = attrs['__init__']
            attrs['__init__'] = _deco(old_init_func)
        return super().__new__(cls, name, bases, attrs)


class Node(metaclass=_NodeMeta):
    """ Node Class """

    @classmethod
    @abc.abstractmethod
    def get_input_keys(cls):
        """ get input keys """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_output_keys(cls):
        """ get output keys """
        raise NotImplementedError

    @classmethod
    def check_input_keys(cls, data):
        """ check input keys """
        required_keys = cls.get_input_keys()
        cls._check_keys(data, required_keys, 'input')

    @classmethod
    def check_output_keys(cls, data):
        """ check output keys """
        required_keys = cls.get_output_keys()
        cls._check_keys(data, required_keys, 'output')

    @classmethod
    def _check_keys(cls, data, required_keys, tag):
        """ check keys """
        if len(required_keys) == 0:
            return
        if isinstance(required_keys[0], list):
            if not all(isinstance(ele, list) for ele in required_keys):
                raise TypeError
            for group in required_keys:
                try:
                    cls._check_keys(data, group, tag)
                except _KeyMissingError:
                    pass
                else:
                    break
            else:
                raise _KeyMissingError(
                    f"The {tag} does not contain the keys required by `{cls.__name__}` object."
                )
        else:
            for key in required_keys:
                if key not in data:
                    raise _KeyMissingError(
                        f"{repr(key)} is a required key in {tag} for `{cls.__name__}` object, but not found."
                    )

    def __repr__(self):
        # TODO: Use fully qualified name which is globally unique
        def _format_args(args_dict):
            """ format arguments
            Refer to https://github.com/albumentations-team/albumentations/blob/\
e3b47b3a127f92541cfeb16abbb44a6f8bf79cc8/albumentations/core/utils.py#L30
            """
            formatted_args = []
            for k, v in args_dict.items():
                if isinstance(v, str):
                    v = f"'{v}'"
                formatted_args.append(f"{k}={v}")
            return ', '.join(formatted_args)

        return '{}({})'.format(self.__class__.__name__,
                               _format_args(getattr(self, '_raw_args', {})))
