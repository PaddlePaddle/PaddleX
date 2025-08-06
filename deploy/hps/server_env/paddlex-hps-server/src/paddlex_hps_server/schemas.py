from importlib import import_module


def __getattr__(name):
    # TODO: Error handling
    return import_module(name=f".{name}", package="paddlex.inference.serving.schemas")
