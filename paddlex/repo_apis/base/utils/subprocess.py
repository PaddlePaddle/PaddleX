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


import asyncio
import subprocess

from ....utils.logging import info

__all__ = ["run_cmd", "CompletedProcess"]


def run_cmd(
    cmd,
    env=None,
    silent=True,
    cwd=None,
    timeout=None,
    echo=False,
    pipe_stdout=False,
    pipe_stderr=False,
    blocking=True,
    async_run=False,
    text=True,
):
    """Wrap around `subprocess.Popen` to execute a shell command."""
    # TODO: Limit argument length
    cfg = dict(env=env, cwd=cwd)

    async_run = async_run and not blocking

    if blocking:
        cfg["timeout"] = timeout
    if silent:
        cfg["stdout"] = (
            subprocess.DEVNULL if not async_run else asyncio.subprocess.DEVNULL
        )
        cfg["stderr"] = (
            subprocess.STDOUT if not async_run else asyncio.subprocess.STDOUT
        )
    if not async_run and (pipe_stdout or pipe_stderr):
        cfg["text"] = True
    if pipe_stdout:
        cfg["stdout"] = subprocess.PIPE if not async_run else asyncio.subprocess.PIPE
    if pipe_stderr:
        cfg["stderr"] = subprocess.PIPE if not async_run else asyncio.subprocess.PIPE

    if echo:
        info(str(cmd))

    if blocking:
        return subprocess.run(cmd, **cfg, check=False)
    else:
        if async_run:
            return asyncio.create_subprocess_exec(cmd[0], *cmd[1:], **cfg)
        else:
            if text:
                cfg.update(dict(bufsize=1, text=True))
            else:
                cfg.update(dict(bufsize=0, text=False))
            return subprocess.Popen(cmd, **cfg)


class CompletedProcess(object):
    """CompletedProcess"""

    __slots__ = ["args", "returncode", "stdout", "stderr", "_add_attrs"]

    def __init__(self, args, returncode, stdout=None, stderr=None):
        super().__init__()
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self._add_attrs = dict()

    def __getattr__(self, name):
        try:
            val = self._add_attrs[name]
            return val
        except KeyError:
            raise AttributeError

    def __setattr__(self, name, val):
        try:
            super().__setattr__(name, val)
        except AttributeError:
            self._add_attrs[name] = val

    def __repr__(self):
        args = [f"args={repr(self.args)}", f"returncode={repr(self.returncode)}"]
        if self.stdout is not None:
            args.append(f"stdout={repr(self.stdout)}")
        if self.stderr is not None:
            args.append(f"stderr={repr(self.stderr)}")
        return f"{self.__class__.__name__}({', '.join(args)})"
