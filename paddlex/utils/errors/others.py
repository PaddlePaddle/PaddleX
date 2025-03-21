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

import json
import signal
from pathlib import Path
from typing import Union

__all__ = [
    "UnsupportedAPIError",
    "UnsupportedParamError",
    "CalledProcessError",
    "raise_unsupported_api_error",
    "raise_key_not_found_error",
    "raise_class_not_found_error",
    "raise_no_entity_registered_error",
    "raise_unsupported_device_error",
    "raise_model_not_found_error",
    "DuplicateRegistrationError",
]


class UnsupportedAPIError(Exception):
    """UnsupportedAPIError"""


class UnsupportedParamError(Exception):
    """UnsupportedParamError"""


class KeyNotFoundError(Exception):
    """KeyNotFoundError"""


class ClassNotFoundException(Exception):
    """ClassNotFoundException"""


class NoEntityRegisteredException(Exception):
    """NoEntityRegisteredException"""


class UnsupportedDeviceError(Exception):
    """UnsupportedDeviceError"""


class CalledProcessError(Exception):
    """CalledProcessError"""

    def __init__(self, returncode, cmd, output=None, stderr=None):
        super().__init__()
        self.returncode = returncode
        self.cmd = cmd
        self.output = output
        self.stderr = stderr

    def __str__(self):
        if self.returncode and self.returncode < 0:
            try:
                return f"Command {repr(self.cmd)} died with {repr(signal.Signals(-self.returncode))}."
            except ValueError:
                return f"Command {repr(self.cmd)} died with unknown signal {-self.returncode}."
        else:
            return f"Command {repr(self.cmd)} returned non-zero exit status {self.returncode}."


class DuplicateRegistrationError(Exception):
    """DuplicateRegistrationError"""


class ModelNotFoundError(Exception):
    """Model Not Found Error"""


def raise_unsupported_api_error(api_name, cls=None):
    """raise unsupported api error"""
    # TODO: Automatically extract `api_name` and `cls` from stack frame
    if cls is not None:
        name = f"{cls.__name__}.{api_name}"
    else:
        name = api_name
    raise UnsupportedAPIError(f"The API `{name}` is not supported.")


def raise_key_not_found_error(key, config=None):
    """raise key not found error"""
    msg = f"`{key}` not found in config."
    if config:
        config_str = json.dumps(config, indent=4, ensure_ascii=False)
        msg += f"\nThe content of config:\n{config_str}"
    raise KeyNotFoundError(msg)


def raise_class_not_found_error(cls_name, base_cls, all_entities=None):
    """raise class not found error"""
    base_cls_name = base_cls.__name__
    msg = f"`{cls_name}` is not registered on {base_cls_name}."
    if all_entities is not None:
        all_entities_str = ",  ".join(all_entities)
        msg += f"\nThe registied entities: [{all_entities_str}]"
    raise ClassNotFoundException(msg)


def raise_no_entity_registered_error(base_cls):
    """raise no entity registered error"""
    base_cls_name = base_cls.__name__
    msg = f"There no entity register on {base_cls_name}. Hint: Maybe the subclass is not imported."
    raise NoEntityRegisteredException(msg)


def raise_unsupported_device_error(device, supported_device=None):
    """raise_unsupported_device_error"""
    msg = f"The device `{device}` is not supported! "
    if supported_device is not None:
        supported_device_str = ", ".join(supported_device)
        msg += f"The supported device types are: [{supported_device_str}]."
    raise UnsupportedDeviceError(msg)


def raise_model_not_found_error(model_path: Union[str, Path]):
    """raise ModelNotFoundError

    Args:
        model_path (str|Path): the path to model file.
    """
    msg = f"The model file(s)(`{model_path}`) is not found."
    raise ModelNotFoundError(msg)
