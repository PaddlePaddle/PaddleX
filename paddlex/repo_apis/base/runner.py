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


import abc
import asyncio
import io
import locale
import os
import shlex
import sys

from ...utils import logging
from ...utils.device import parse_device
from ...utils.errors import CalledProcessError, raise_unsupported_api_error
from ...utils.flags import DRY_RUN
from ...utils.misc import abspath
from .utils.arg import CLIArgument
from .utils.subprocess import CompletedProcess
from .utils.subprocess import run_cmd as _run_cmd

__all__ = ["BaseRunner", "InferOnlyRunner"]


class BaseRunner(metaclass=abc.ABCMeta):
    """
    Abstract base class of Runner.

    Runner is responsible for executing training/inference/compression commands.
    """

    def __init__(self, runner_root_path):
        """
        Initialize the instance.

        Args:
            runner_root_path (str): Path of the directory where the scripts reside.
        """
        super().__init__()
        self.runner_root_path = abspath(runner_root_path)
        # Path to python interpreter
        self.python = sys.executable

    def prepare(self):
        """
        Make preparations for the execution of commands.

        For example, download prerequisites and install dependencies.
        """
        # By default we do nothing

    @abc.abstractmethod
    def train(self, config_path, cli_args, device, ips, save_dir, do_eval=True):
        """
        Execute model training command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[base.utils.arg.CLIArgument]): List of command-line
                arguments.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.
            ips (str|None): Paddle cluster node ips, e.g.,
                '192.168.0.16,192.168.0.17'.
            save_dir (str): Directory to save log files.
            do_eval (bool, optional): Whether to perform model evaluation during
                training. Default: True.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, config_path, cli_args, device, ips):
        """
        Execute model evaluation command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[base.utils.arg.CLIArgument]): List of command-line
                arguments.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.
            ips (str|None): Paddle cluster node ips, e.g.,
                '192.168.0.16,192.168.0.17'.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, config_path, cli_args, device):
        """
        Execute prediction command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[base.utils.arg.CLIArgument]): List of command-line
                arguments.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def export(self, config_path, cli_args, device):
        """
        Execute model export command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[base.utils.arg.CLIArgument]): List of command-line
                arguments.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def infer(self, config_path, cli_args, device):
        """
        Execute model inference command.

        Args:
            config_path (str): Path of the configuration file.
            cli_args (list[base.utils.arg.CLIArgument]): List of command-line
                arguments.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.

        Returns:
            paddlex.repo_apis.base.utils.subprocess.CompletedProcess
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compression(
        self, config_path, train_cli_args, export_cli_args, device, train_save_dir
    ):
        """
        Execute model compression (quantization aware training and model export)
            commands.

        Args:
            config_path (str): Path of the configuration file.
            train_cli_args (list[base.utils.arg.CLIArgument]): List of
                command-line arguments used for model training.
            export_cli_args (list[base.utils.arg.CLIArgument]): List of
                command-line arguments used for model export.
            device (str): A string that describes the device(s) to use, e.g.,
                'cpu', 'xpu:0', 'gpu:1,2'.
            train_save_dir (str): Directory to store model snapshots.

        Returns:
            tuple[paddlex.repo_apis.base.utils.subprocess.CompletedProcess]
        """
        raise NotImplementedError

    def distributed(self, device, ips=None, log_dir=None):
        """distributed"""
        # TODO: docstring
        args = [self.python]
        if device is None:
            return args, None
        device, dev_ids = parse_device(device)
        if dev_ids is None or len(dev_ids) == 0:
            return args, None
        else:
            num_devices = len(dev_ids)
            dev_ids = ",".join([str(n) for n in dev_ids])
        if num_devices > 1:
            args.extend(["-m", "paddle.distributed.launch"])
            args.extend(["--devices", dev_ids])
            if ips is not None:
                args.extend(["--ips", ips])
            if log_dir is None:
                log_dir = os.getcwd()
            args.extend(["--log_dir", self._get_dist_train_log_dir(log_dir)])
        elif num_devices == 1:
            new_env = os.environ.copy()
            if device == "xpu":
                new_env["XPU_VISIBLE_DEVICES"] = dev_ids
            elif device == "npu":
                new_env["ASCEND_RT_VISIBLE_DEVICES"] = dev_ids
            elif device == "mlu":
                new_env["MLU_VISIBLE_DEVICES"] = dev_ids
            elif device == "gcu":
                new_env["TOPS_VISIBLE_DEVICES"] = dev_ids
            else:
                new_env["CUDA_VISIBLE_DEVICES"] = dev_ids
            return args, new_env
        return args, None

    def run_cmd(
        self,
        cmd,
        env=None,
        switch_wdir=True,
        silent=False,
        echo=True,
        capture_output=False,
        log_path=None,
    ):
        """run_cmd"""

        def _trans_args(cmd):
            out = []
            for ele in cmd:
                if isinstance(ele, CLIArgument):
                    out.extend(ele.lst)
                else:
                    out.append(ele)
            return out

        cmd = _trans_args(cmd)

        if DRY_RUN:
            # TODO: Accommodate Windows system
            logging.info(" ".join(shlex.quote(x) for x in cmd))
            # Mock return
            return CompletedProcess(cmd, returncode=0, stdout=str(cmd), stderr=None)

        if switch_wdir:
            if isinstance(switch_wdir, str):
                # In this case `switch_wdir` specifies a relative path
                cwd = os.path.join(self.runner_root_path, switch_wdir)
            else:
                cwd = self.runner_root_path
        else:
            cwd = None

        if not capture_output:
            if log_path is not None:
                logging.warning(
                    "`log_path` will be ignored when `capture_output` is False."
                )

            cp = _run_cmd(
                cmd,
                env=env,
                cwd=cwd,
                silent=silent,
                echo=echo,
                pipe_stdout=False,
                pipe_stderr=False,
                blocking=True,
            )
            cp = CompletedProcess(
                cp.args, cp.returncode, stdout=cp.stdout, stderr=cp.stderr
            )
        else:
            # Refer to
            # https://stackoverflow.com/questions/17190221/subprocess-popen-cloning-stdout-and-stderr-both-to-terminal-and-variables/25960956
            async def _read_display_and_record_from_stream(
                in_stream, out_stream, files
            ):
                # According to
                # https://docs.python.org/3/library/subprocess.html#frequently-used-arguments
                _ENCODING = locale.getpreferredencoding(False)
                chars = []
                out_stream_is_buffered = hasattr(out_stream, "buffer")
                while True:
                    flush = False
                    char = await in_stream.read(1)
                    if char == b"":
                        break
                    if out_stream_is_buffered:
                        out_stream.buffer.write(char)
                    chars.append(char)
                    if char == b"\n":
                        flush = True
                    elif char == b"\r":
                        # NOTE: In order to get tqdm progress bars to produce normal outputs
                        # we treat '\r' as an ending character of line
                        flush = True
                    if flush:
                        line = b"".join(chars)
                        line = line.decode(_ENCODING)
                        if not out_stream_is_buffered:
                            # We use line buffering
                            out_stream.write(line)
                        else:
                            out_stream.buffer.flush()
                        for f in files:
                            f.write(line)
                        chars.clear()

            async def _tee_proc_call(proc_call, out_files, err_files):
                proc = await proc_call
                await asyncio.gather(
                    _read_display_and_record_from_stream(
                        proc.stdout, sys.stdout, out_files
                    ),
                    _read_display_and_record_from_stream(
                        proc.stderr, sys.stderr, err_files
                    ),
                )
                # NOTE: https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
                retcode = await proc.wait()
                return retcode

            # Non-blocking call with stdout and stderr piped
            with io.StringIO() as stdout_buf, io.StringIO() as stderr_buf:
                proc_call = _run_cmd(
                    cmd,
                    env=env,
                    cwd=cwd,
                    echo=echo,
                    silent=silent,
                    pipe_stdout=True,
                    pipe_stderr=True,
                    blocking=False,
                    async_run=True,
                )
                out_files = [stdout_buf]
                err_files = [stderr_buf]
                if log_path is not None:
                    log_dir = os.path.dirname(log_path)
                    os.makedirs(log_dir, exist_ok=True)
                    log_file = open(log_path, "w", encoding="utf-8")
                    logging.info(f"\nLog path: {os.path.abspath(log_path)} \n")
                    out_files.append(log_file)
                    err_files.append(log_file)
                try:
                    retcode = asyncio.run(
                        _tee_proc_call(proc_call, out_files, err_files)
                    )
                finally:
                    if log_path is not None:
                        log_file.close()
                cp = CompletedProcess(
                    cmd, retcode, stdout_buf.getvalue(), stderr_buf.getvalue()
                )

        if cp.returncode != 0:
            raise CalledProcessError(
                cp.returncode, cp.args, output=cp.stdout, stderr=cp.stderr
            )
        return cp

    def _get_dist_train_log_dir(self, log_dir):
        """_get_dist_train_log_dir"""
        return os.path.join(log_dir, "distributed_train_logs")

    def _get_train_log_path(self, log_dir):
        """_get_train_log_path"""
        return os.path.join(log_dir, "train.log")


class InferOnlyRunner(BaseRunner):
    """InferOnlyRunner"""

    def train(self, *args, **kwargs):
        """train"""
        raise_unsupported_api_error(self.__class__, "train")

    def evaluate(self, *args, **kwargs):
        """evaluate"""
        raise_unsupported_api_error(self.__class__, "evalaute")

    def predict(self, *args, **kwargs):
        """predict"""
        raise_unsupported_api_error(self.__class__, "predict")

    def export(self, *args, **kwargs):
        """export"""
        raise_unsupported_api_error(self.__class__, "export")

    def compression(self, *args, **kwargs):
        """compression"""
        raise_unsupported_api_error(self.__class__, "compression")
