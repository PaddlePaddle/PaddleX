from importlib import metadata as _metadata

from .base_model import BaseTritonPythonModel
from .logging import set_up_logger as _set_up_logger

__all__ = ["__version__", "BaseTritonPythonModel"]

# Ref: https://github.com/langchain-ai/langchain/blob/493e474063817b9a4c2521586b2dbc34d20b4cf1/libs/core/langchain_core/__init__.py
try:
    __version__ = _metadata.version(__package__)
except _metadata.PackageNotFoundError:
    __version__ = ""

_set_up_logger()
