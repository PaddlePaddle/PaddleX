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
from types import GeneratorType


def batchable_method(func):
    """batchable"""

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
        raise TypeError("The function to wrap should have at least two parameters.")
    return _wrapper


def generatorable_method(func):
    """generatorable"""

    @functools.wraps(func)
    def _wrapper(self, input_, *args, **kwargs):
        if isinstance(input_, GeneratorType):
            for ele in input_:
                yield func(self, ele, *args, **kwargs)
        else:
            yield func(self, input_, *args, **kwargs)

    sig = inspect.signature(func)
    if not len(sig.parameters) >= 2:
        raise TypeError("The function to wrap should have at least two parameters.")
    return _wrapper
