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


__all__ = [
    "build_res_dict",
    "CheckFailedError",
    "ConvertFailedError",
    "SplitFailedError",
    "AnalyseFailedError",
    "DatasetFileNotFoundError",
]


def build_res_dict(check_pass, err_type=None, err_msg=None, **kwargs):
    """build the dataset checking result to return"""
    if check_pass:
        if err_type is not None:
            raise ValueError(
                f"`check_pass` is {check_pass}, but `err_type` is not None."
            )
        if err_msg is not None:
            raise ValueError(
                f"`check_pass` is {check_pass}, but `err_msg` is not None."
            )
        return dict(check_pass=check_pass, **kwargs)
    else:
        if err_type is None:
            raise ValueError(f"`check_pass` is {check_pass}, but `err_type` is None.")
        if err_msg is None:
            if _is_known_error_type(err_type):
                err_msg = ""
            else:
                raise ValueError(
                    f"{err_type} is not a known error type, in which case `err_msg` must be specified to a value other \
than None."
                )
        return dict(check_pass=check_pass, err_type=err_type, err_msg=err_msg, **kwargs)


class FailedError(Exception):
    """base error class"""

    def __init__(self, err_info=None, solution=None, message=None):
        if message is None:
            message = self._construct_message(err_info, solution)
        super().__init__(message)

    def _construct_message(self, err_info, solution):
        if err_info is None:
            return ""
        else:
            msg = (
                f"{self.mode} failed. We encountered the following error:\n  {err_info}"
            )
            if solution is not None:
                msg += f"\nPlease try to resolve the issue as follows:\n  {solution}"
            return msg


class CheckFailedError(FailedError):
    """check dataset error"""

    mode = "Check dataset"


class ConvertFailedError(FailedError):
    """convert dataset error"""

    mode = "Convert dataset"


class SplitFailedError(FailedError):
    """split dataset error"""

    mode = "Split dataset"


class AnalyseFailedError(FailedError):
    """analyse dataset error"""

    mode = "Analyse dataset"


class DatasetFileNotFoundError(CheckFailedError):
    """dataset file not found error"""

    def __init__(self, file_path=None, err_info=None, solution=None, message=None):
        if err_info is None:
            if file_path is not None:
                err_info = f"{file_path} does not exist."
        super().__init__(err_info, solution, message)


def _is_known_error_type(err_type):
    """is known error type"""
    return isinstance(err_type, CheckFailedError)
