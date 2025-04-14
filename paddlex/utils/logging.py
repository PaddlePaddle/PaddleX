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


import functools
import inspect
import logging
import sys

import colorlog

from .flags import DEBUG

__all__ = ["debug", "info", "warning", "error", "critical", "setup_logging"]

LOGGER_NAME = "paddlex"
_LOG_CONFIG = {
    "DEBUG": {"color": "purple"},
    "INFO": {"color": "green"},
    "WARNING": {"color": "yellow"},
    "ERROR": {"color": "red"},
    "CRITICAL": {"color": "bold_red"},
}

_logger = logging.getLogger(LOGGER_NAME)


def debug(msg, *args, **kwargs):
    """debug"""
    if DEBUG:
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_func_name = caller_frame.f_code.co_name

        if "self" in caller_frame.f_locals:
            caller_class_name = caller_frame.f_locals["self"].__class__.__name__
        elif "cls" in caller_frame.f_locals:
            caller_class_name = caller_frame.f_locals["cls"].__name__
        else:
            caller_class_name = None

        if caller_class_name:
            caller_info = f"{caller_class_name}::{caller_func_name}"
        else:
            caller_info = f"{caller_func_name}"
        msg = f"【{caller_info}】{msg}"
        _logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """info"""
    _logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """warning"""
    _logger.warning(msg, *args, **kwargs)


@functools.lru_cache(None)
def warning_once(msg, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once
    """
    warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """error"""
    _logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """critical"""
    _logger.critical(msg, *args, **kwargs)


def exception(msg, *args, **kwargs):
    """exception"""
    _logger.exception(msg, *args, **kwargs)


def setup_logging(verbosity: str = None):
    """setup logging level

    Args:
        verbosity (str, optional): the logging level, `DEBUG`, `INFO`, `WARNING`. Defaults to None.
    """
    if verbosity is None:
        if DEBUG:
            verbosity = "DEBUG"
        else:
            verbosity = "INFO"

    if verbosity is not None:
        _configure_logger(_logger, verbosity.upper())


def _configure_logger(logger, verbosity):
    """_configure_logger"""
    if verbosity == "DEBUG":
        _logger.setLevel(logging.DEBUG)
    elif verbosity == "INFO":
        _logger.setLevel(logging.INFO)
    elif verbosity == "WARNING":
        _logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.hasHandlers():
        _add_handler(logger)


def _add_handler(logger):
    """_add_handler"""
    format = colorlog.ColoredFormatter(
        "%(log_color)s%(message)s",
        log_colors={key: conf["color"] for key, conf in _LOG_CONFIG.items()},
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(format)
    logger.addHandler(handler)


def advertise():
    """
    Show the advertising message like the following:

    ===========================================================
    ==         PaddleX is powered by PaddlePaddle !          ==
    ===========================================================
    ==                                                       ==
    ==   For more info please go to the following website.   ==
    ==                                                       ==
    ==        https://github.com/PaddlePaddle/PaddleX        ==
    ===========================================================

    """
    copyright = "PaddleX is powered by PaddlePaddle !"
    ad = "For more info please go to the following website."
    website = "https://github.com/PaddlePaddle/PaddleX"
    AD_LEN = 6 + len(max([copyright, ad, website], key=len))

    info(
        "\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
            "=" * (AD_LEN + 4),
            "=={}==".format(copyright.center(AD_LEN)),
            "=" * (AD_LEN + 4),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(ad.center(AD_LEN)),
            "=={}==".format(" " * AD_LEN),
            "=={}==".format(website.center(AD_LEN)),
            "=" * (AD_LEN + 4),
        )
    )
