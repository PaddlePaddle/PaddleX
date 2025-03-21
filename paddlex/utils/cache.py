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


import atexit
import functools
import hashlib
import inspect
import os
import os.path as osp
import pickle
import tempfile
from pathlib import Path

import filelock

DEFAULT_CACHE_DIR = osp.abspath(osp.join(os.path.expanduser("~"), ".paddlex"))
CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", DEFAULT_CACHE_DIR)
FUNC_CACHE_DIR = osp.join(CACHE_DIR, "func_ret")
FILE_LOCK_DIR = osp.join(CACHE_DIR, "locks")
TEMP_DIR = osp.join(CACHE_DIR, "temp")


def create_cache_dir(*args, **kwargs):
    """create cache dir"""
    # `args` and `kwargs` reserved for extension
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(FUNC_CACHE_DIR, exist_ok=True)
    os.makedirs(FILE_LOCK_DIR, exist_ok=True)
    # TODO: Ensure permission


def get_cache_dir(*args, **kwargs):
    """get cache dir"""
    # `args` and `kwargs` reserved for extension
    return CACHE_DIR


def persist(cond=None):
    """persist"""
    # FIXME: Current implementation creates files in cache dir and we do
    # not set a limit on number of files
    # TODO: Faster implementation and support more arg types
    FILENAME_PATTERN = "persist_{key}.pkl"
    SUPPORTED_ARG_TYPES = (str, int, float)

    if cond is None:
        cond = lambda ret: ret is not None

    def _to_bytes(obj):
        return str(obj).encode("utf-8")

    def _make_key(func, bnd_args):
        # Use MD5 algorithm to make deterministic hashing
        # Note that the object-to-bytes conversion should be deterministic,
        # we ensure this by restricting types of arguments.
        m = hashlib.md5()
        m.update(_to_bytes(osp.realpath(inspect.getsourcefile(func))))
        m.update(_to_bytes(func.__name__))
        for k, v in bnd_args.arguments.items():
            if not isinstance(v, SUPPORTED_ARG_TYPES):
                raise TypeError(
                    f"{repr(k)}: {v}, {type(v)} is unhashable or not a supported type."
                )
            m.update(_to_bytes(k))
            m.update(_to_bytes(v))
        hash_ = m.hexdigest()
        return hash_

    def _deco(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bnd_args = sig.bind(*args, **kwargs)
            bnd_args.apply_defaults()
            key = _make_key(func, bnd_args)
            cache_file_path = osp.join(
                FUNC_CACHE_DIR, FILENAME_PATTERN.format(key=str(key))
            )
            lock = filelock.FileLock(osp.join(FILE_LOCK_DIR, f"{key}.lock"))
            with lock:
                if osp.exists(cache_file_path):
                    with open(cache_file_path, "rb") as f:
                        ret = pickle.load(f)
                else:
                    ret = func(*args, **kwargs)
                    if cond(ret):
                        with open(cache_file_path, "wb") as f:
                            pickle.dump(ret, f)
            return ret

        return _wrapper

    return _deco


class TempFileManager:
    def __init__(self):
        self.temp_files = []
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        atexit.register(self.cleanup)

    def create_temp_file(self, **kwargs):
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=TEMP_DIR, **kwargs)
        self.temp_files.append(temp_file)
        return temp_file

    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                temp_file.close()
                os.remove(temp_file.name)
            except FileNotFoundError:
                pass
        self.temp_files = []

    class TempFileContextManager:
        def __init__(self, manager, **kwargs):
            self.manager = manager
            self.kwargs = kwargs
            self.temp_file = None

        def __enter__(self):
            self.temp_file = self.manager.create_temp_file(**self.kwargs)
            return self.temp_file

        def __exit__(self, exc_type, exc_value, traceback):
            if self.temp_file:
                self.temp_file.close()

    def temp_file_context(self, **kwargs):
        return self.TempFileContextManager(self, **kwargs)


temp_file_manager = TempFileManager()
