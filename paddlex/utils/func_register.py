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

from functools import wraps

from . import logging


class FuncRegister(object):
    def __init__(self, register_map):
        assert isinstance(register_map, dict)
        self._register_map = register_map

    def __call__(self, key):
        """register the decoratored func as key in dict"""

        def decorator(func):
            self._register_map[key] = func
            logging.debug(
                f"The func ({func.__name__}) has been registered as key ({key})."
            )

            @wraps(func)
            def wrapper(self, *args, **kwargs):
                return func(self, *args, **kwargs)

            return wrapper

        return decorator
