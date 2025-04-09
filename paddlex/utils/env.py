# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import os
from contextlib import ContextDecorator


def get_device_type():
    import paddle

    device_str = paddle.get_device()
    return device_str.split(":")[0]


def get_paddle_version():
    import paddle

    version = paddle.__version__.split(".")
    # ref: https://github.com/PaddlePaddle/Paddle/blob/release/3.0-beta2/setup.py#L316
    assert len(version) == 3
    major_v, minor_v, patch_v = version
    return major_v, minor_v, patch_v


class TemporaryEnvVarChanger(ContextDecorator):
    """
    A context manager to temporarily change an environment variable's value
    and restore it upon exiting the context.
    """

    def __init__(self, var_name, new_value):
        # if new_value is None, nothing changed
        self.var_name = var_name
        self.new_value = new_value
        self.original_value = None

    def __enter__(self):
        if self.new_value is None:
            return self
        self.original_value = os.environ.get(self.var_name, None)
        os.environ[self.var_name] = self.new_value
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.new_value is None:
            return False
        if self.original_value is not None:
            os.environ[self.var_name] = self.original_value
        elif self.var_name in os.environ:
            del os.environ[self.var_name]
        return False
