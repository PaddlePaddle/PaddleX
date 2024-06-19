# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
