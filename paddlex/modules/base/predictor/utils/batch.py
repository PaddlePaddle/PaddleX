# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import inspect
import functools
import itertools

__all__ = ['batchable_method', 'apply_batch', 'Batcher']


def batchable_method(func):
    """ batchable """

    @functools.wraps(func)
    def _wrapper(self, input_, *args, **kwargs):
        if isinstance(input_, list):
            output = []
            for ele in input_:
                out = func(self, ele, *args, **kwargs)
                output.append(out)
            return output
        else:
            return func(self, input_, *args, **kwargs)

    sig = inspect.signature(func)
    if not len(sig.parameters) >= 2:
        raise TypeError(
            "The function to wrap should have at least two parameters.")
    return _wrapper


def apply_batch(batch, callable_, *args, **kwargs):
    """ apply batch  """
    output = []
    for ele in batch:
        out = callable_(ele, *args, **kwargs)
        output.append(out)
    return output


class Batcher(object):
    """ Batcher """

    def __init__(self, iterable, batch_size=None):
        super().__init__()
        self.iterable = iterable
        self.batch_size = batch_size

    def __iter__(self):
        if self.batch_size is None:
            all_data = list(self.iterable)
            yield all_data
        else:
            iterator = iter(self.iterable)
            while True:
                batch = list(itertools.islice(iterator, self.batch_size))
                if not batch:
                    break
                yield batch
