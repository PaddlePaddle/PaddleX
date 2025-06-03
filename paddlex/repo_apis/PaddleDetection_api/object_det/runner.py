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


import os

from ...base import BaseRunner
from ...base.utils.arg import CLIArgument, gather_opts_args
from ...base.utils.subprocess import CompletedProcess


class DetRunner(BaseRunner):
    """DetRunner"""

    def train(
        self,
        config_path: str,
        cli_args: list,
        device: str,
        ips: str,
        save_dir: str,
        do_eval=True,
    ) -> CompletedProcess:
        """train model

        Args:
            config_path (str): the config file path used to train.
            cli_args (list): the additional parameters.
            device (str): the training device.
            ips (str): the ip addresses of nodes when using distribution.
            save_dir (str): the directory path to save training output.
            do_eval (bool, optional): whether or not to evaluate model during training. Defaults to True.

        Returns:
            CompletedProcess: the result of training subprocess execution.
        """
        args, env = self.distributed(device, ips, log_dir=save_dir)
        cli_args = self._gather_opts_args(cli_args)
        cmd = [*args, "tools/train.py"]
        if do_eval:
            cmd.append("--eval")
        cmd.extend(["--config", config_path, *cli_args])
        return self.run_cmd(
            cmd,
            env=env,
            switch_wdir=True,
            echo=True,
            silent=False,
            capture_output=True,
            log_path=self._get_train_log_path(save_dir),
        )

    def evaluate(
        self, config_path: str, cli_args: list, device: str, ips: str
    ) -> CompletedProcess:
        """run model evaluating

        Args:
            config_path (str): the config file path used to evaluate.
            cli_args (list): the additional parameters.
            device (str): the evaluating device.
            ips (str): the ip addresses of nodes when using distribution.

        Returns:
            CompletedProcess: the result of evaluating subprocess execution.
        """
        args, env = self.distributed(device, ips)
        cli_args = self._gather_opts_args(cli_args)
        cmd = [*args, "tools/eval.py", "--config", config_path, *cli_args]
        cp = self.run_cmd(
            cmd, env=env, switch_wdir=True, echo=True, silent=False, capture_output=True
        )
        if cp.returncode == 0:
            metric_dict = _extract_eval_metrics(cp.stdout)
            cp.metrics = metric_dict
        return cp

    def predict(
        self, config_path: str, cli_args: list, device: str
    ) -> CompletedProcess:
        """run predicting using dynamic mode

        Args:
            config_path (str): the config file path used to predict.
            cli_args (list): the additional parameters.
            device (str): unused.

        Returns:
            CompletedProcess: the result of predicting subprocess execution.
        """
        # `device` unused
        cli_args = self._gather_opts_args(cli_args)
        cmd = [self.python, "tools/infer.py", "-c", config_path, *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path: str, cli_args: list, device: str) -> CompletedProcess:
        """run exporting

        Args:
            config_path (str): the path of config file used to export.
            cli_args (list): the additional parameters.
            device (str): unused.
            save_dir (str, optional): the directory path to save exporting output. Defaults to None.

        Returns:
            CompletedProcess: the result of exporting subprocess execution.
        """
        # `device` unused
        cli_args = self._gather_opts_args(cli_args)
        cmd = [
            self.python,
            "tools/export_model.py",
            "--for_fd",
            "-c",
            config_path,
            *cli_args,
        ]

        cp = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
        return cp

    def infer(self, cli_args: list, device: str) -> CompletedProcess:
        """run predicting using inference model

        Args:
            cli_args (list): the additional parameters.
            device (str): unused.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        # `device` unused
        cmd = [self.python, "deploy/python/infer.py", "--use_fd_format", *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def compression(
        self,
        config_path: str,
        train_cli_args: list,
        export_cli_args: list,
        device: str,
        train_save_dir: str,
    ) -> CompletedProcess:
        """run compression model

        Args:
            config_path (str): the path of config file used to predict.
            train_cli_args (list): the additional training parameters.
            export_cli_args (list): the additional exporting parameters.
            device (str): the running device.
            train_save_dir (str): the directory path to save output.

        Returns:
            CompletedProcess: the result of compression subprocess execution.
        """
        args, env = self.distributed(device, log_dir=train_save_dir)
        train_cli_args = self._gather_opts_args(train_cli_args)
        cmd = [*args, "tools/train.py", "-c", config_path, *train_cli_args]
        cp_train = self.run_cmd(
            cmd,
            env=env,
            switch_wdir=True,
            echo=True,
            silent=False,
            capture_output=True,
            log_path=self._get_train_log_path(train_save_dir),
        )

        cps_weight_path = os.path.join(train_save_dir, "model_final")

        export_cli_args.append(CLIArgument("-o", f"weights={cps_weight_path}"))
        export_cli_args = self._gather_opts_args(export_cli_args)
        cmd = [
            self.python,
            "tools/export_model.py",
            "--for_fd",
            "-c",
            config_path,
            *export_cli_args,
        ]
        cp_export = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        return cp_train, cp_export

    def _gather_opts_args(self, args):
        """_gather_opts_args"""
        return gather_opts_args(args, "-o")


def _extract_eval_metrics(stdout):
    """extract evaluation metrics from training log

    Args:
        stdout (str): the training log

    Returns:
        dict: the training metric
    """
    import re

    pattern = r".*\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s\|\smaxDets=\s*\d+\s\]\s*=\s*[0-1]?\.[0-9]{3}$"
    key = "AP"

    metric_dict = dict()
    pattern = re.compile(pattern)
    # TODO: Use lazy version to make it more efficient
    lines = stdout.splitlines()
    metric_dict[key] = 0
    for line in lines:
        match = pattern.search(line)
        if match:
            metric_dict[key] = float(match.group(0)[-5:])

    return metric_dict
