import importlib


class _LazyModule(object):
    def __init__(self, mod_name):
        super().__init__()
        self.mod_name = mod_name
        self._mod = None

    def __getattr__(self, name):
        if not self._mod:
            self._mod = importlib.import_module(self.mod_name)
        return getattr(self._mod, name)


pb_utils = _LazyModule("triton_python_backend_utils")
