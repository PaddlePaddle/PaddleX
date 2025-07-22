import logging
import sys
from contextvars import ContextVar

import colorlog

from . import env

_LOGGING_CONFIG = {
    "DEBUG": {"color": "purple"},
    "INFO": {"color": "green"},
    "WARNING": {"color": "yellow"},
    "ERROR": {"color": "red"},
    "CRITICAL": {"color": "bold_red"},
}

model_id_var = ContextVar("model_id", default="*")
log_id_var = ContextVar("log_id", default="*")
_logger = logging.getLogger("paddlex-hps-server")


def _log_with_context(func):
    def _wrapper(msg, *args, **kwargs):
        extra = kwargs.get("extra", {})
        extra["model_id"] = model_id_var.get()
        extra["log_id"] = log_id_var.get()
        kwargs["extra"] = extra
        return func(msg, *args, **kwargs)

    return _wrapper


def set_up_logger():
    if env.LOGGING_LEVEL:
        _logger.setLevel(env.LOGGING_LEVEL)
        format = colorlog.ColoredFormatter(
            "%(log_color)s[%(levelname)8s] [%(asctime)-15s] [%(model_id)s] [%(log_id)s] - %(message)s",
            log_colors={key: conf["color"] for key, conf in _LOGGING_CONFIG.items()},
        )
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(format)
        _logger.addHandler(handler)
        _logger.propagate = False


def set_context_vars(model_id, log_id):
    return model_id_var.set(model_id), log_id_var.set(log_id)


def reset_context_vars(model_id_token, log_id_token):
    model_id_var.reset(model_id_token)
    log_id_var.reset(log_id_token)


debug = _log_with_context(_logger.debug)
info = _log_with_context(_logger.info)
warning = _log_with_context(_logger.warning)
error = _log_with_context(_logger.error)
critical = _log_with_context(_logger.critical)
