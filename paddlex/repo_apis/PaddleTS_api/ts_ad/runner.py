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


from ....utils.errors import raise_unsupported_api_error
from ...base import BaseRunner
from ...base.utils.arg import gather_opts_args
from ...base.utils.subprocess import CompletedProcess


class TSADRunner(BaseRunner):
    """TS Anomaly Detection Runner"""

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
        cmd = [*args, "tools/train.py", "--config", config_path, *cli_args]
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
        cmd = [*args, "tools/val.py", "--config", config_path, *cli_args]

        cp = self.run_cmd(
            cmd, env=env, switch_wdir=True, echo=True, silent=False, capture_output=True
        )
        if cp.returncode == 0:
            metric_dict = _extract_eval_metrics(cp.stderr)
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
        cmd = [self.python, "tools/predict.py", "--config", config_path, *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path, cli_args, device):
        """export"""
        cmd = [
            self.python,
            "tools/export.py",
            "--config",
            config_path,
            *cli_args,
        ]
        cp = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
        return cp

    def infer(self, config_path, cli_args, device):
        """infer"""
        raise_unsupported_api_error("infer", self.__class__)

    def compression(
        self, config_path, train_cli_args, export_cli_args, device, train_save_dir
    ):
        """compression"""
        raise_unsupported_api_error("compression", self.__class__)

    def _gather_opts_args(self, args):
        # Since `--opts` in PaddleSeg does not use `action='append'`
        # We collect and arrange all opts args here
        # e.g.: python tools/train.py --config xxx --opts a=1 c=3 --opts b=2
        # => python tools/train.py --config xxx c=3 --opts a=1 b=2
        return gather_opts_args(args, "--opts")


def _extract_eval_metrics(stdout: str) -> dict:
    """extract evaluation metrics from training log

    Args:
        stdout (str): the training log

    Returns:
        dict: the training metric
    """
    import re

    pattern = r"\'f1\':\s+(\d+\.\d+),+[\s|\n]+\'precision\':\s+(\d+\.\d+),+[\s|\n]+\'recall\':\s+(\d+\.\d+)"
    keys = ["f1", "precision", "recall"]

    metric_dict = dict()
    pattern = re.compile(pattern)

    lines = stdout.splitlines()
    for line in lines:
        match = pattern.search(line)
        if match:
            for k, v in zip(keys, map(float, match.groups())):
                metric_dict[k] = v

    return metric_dict
