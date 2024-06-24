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

import logging
import sys

import colorlog

from .flags import DEBUG

__all__ = ['debug', 'info', 'warning', 'error', 'critical', 'setup_logging']

LOGGER_NAME = 'paddlex'
_LOG_CONFIG = {
    'DEBUG': {
        'color': 'purple'
    },
    'INFO': {
        'color': 'green'
    },
    'WARNING': {
        'color': 'yellow'
    },
    'ERROR': {
        'color': 'red'
    },
    'CRITICAL': {
        'color': 'bold_red'
    },
}

_logger = logging.getLogger(LOGGER_NAME)


def debug(msg, *args, **kwargs):
    """ debug """
    _logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    """ info """
    _logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """ warning """
    _logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """ error """
    _logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """ critical """
    _logger.critical(msg, *args, **kwargs)


def setup_logging(verbosity: str=None):
    """setup logging level

    Args:
        verbosity (str, optional): the logging level, `DEBUG`, `INFO`, `WARNING`. Defaults to None.
    """
    if verbosity is None:
        if DEBUG:
            verbosity = 'DEBUG'
        else:
            verbosity = 'INFO'

    if verbosity is not None:
        _configure_logger(_logger, verbosity.upper())


def _configure_logger(logger, verbosity):
    """ _configure_logger """
    if verbosity == 'DEBUG':
        _logger.setLevel(logging.DEBUG)
    elif verbosity == 'INFO':
        _logger.setLevel(logging.INFO)
    elif verbosity == 'WARNING':
        _logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not logger.hasHandlers():
        _add_handler(logger)


def _add_handler(logger):
    """ _add_handler """
    format = colorlog.ColoredFormatter(
        '%(log_color)s%(message)s',
        log_colors={key: conf['color']
                    for key, conf in _LOG_CONFIG.items()}, )

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

    info("\n{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n".format(
        "=" * (AD_LEN + 4),
        "=={}==".format(copyright.center(AD_LEN)),
        "=" * (AD_LEN + 4),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(ad.center(AD_LEN)),
        "=={}==".format(' ' * AD_LEN),
        "=={}==".format(website.center(AD_LEN)),
        "=" * (AD_LEN + 4), ))
