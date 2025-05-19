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
from ...base.utils.subprocess import CompletedProcess


class ClsRunner(BaseRunner):
    """Cls Runner"""

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
        cmd = [*args, "tools/train.py", "-c", config_path, *cli_args]
        cmd.extend(["-o", f"Global.eval_during_train={do_eval}"])
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
        cmd = [*args, "tools/eval.py", "-c", config_path, *cli_args]
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
        cmd = [self.python, "tools/infer.py", "-c", config_path, *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(
        self, config_path: str, cli_args: list, device: str, save_dir: str = None
    ) -> CompletedProcess:
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

        cmd = [
            self.python,
            "tools/export_model.py",
            "-c",
            config_path,
            *cli_args,
            "-o",
            "Global.export_for_fd=True",
        ]

        cp = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
        return cp

    def infer(self, config_path: str, cli_args: list, device: str) -> CompletedProcess:
        """run predicting using inference model

        Args:
            config_path (str): the path of config file used to predict.
            cli_args (list): the additional parameters.
            device (str): unused.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        # `device` unused
        cmd = [self.python, "python/predict_cls.py", "-c", config_path, *cli_args]
        return self.run_cmd(cmd, switch_wdir="deploy", echo=True, silent=False)

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
        # Step 1: Train model
        cp_train = self.train(config_path, train_cli_args, device, None, train_save_dir)

        # Step 2: Export model
        weight_path = os.path.join(train_save_dir, "best_model", "model")
        export_cli_args = [
            *export_cli_args,
            "-o",
            f"Global.pretrained_model={weight_path}",
        ]
        cp_export = self.export(config_path, export_cli_args, device)

        return cp_train, cp_export


def _extract_eval_metrics(stdout: str) -> dict:
    """extract evaluation metrics from training log

    Args:
        stdout (str): the training log

    Returns:
        dict: the training metric
    """
    import re

    _DP = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    patterns = [
        r"\[Eval\]\[Epoch 0\]\[Avg\].*top1: (_dp)".replace("_dp", _DP),
        r"\[Eval\]\[Epoch 0\]\[Avg\].*top1: (_dp), top5: (_dp)".replace("_dp", _DP),
        r"\[Eval\]\[Epoch 0\]\[Avg\].*recall1: (_dp), recall5: (_dp), mAP: (_dp)".replace(
            "_dp", _DP
        ),
        r"\[Eval\]\[Epoch 0\]\[Avg\].*MultiLabelMAP\(integral\): (_dp)".replace(
            "_dp", _DP
        ),
        r"\[Eval\]\[Epoch 0\]\[Avg\].*evalres:\ ma: (_dp)".replace("_dp", _DP),
    ]
    keys = [
        ["val.top1"],
        ["val.top1", "val.top5"],
        ["recall1", "recall5", "mAP"],
        ["MultiLabelMAP"],
        ["evalres: ma"],
    ]

    metric_dict = dict()
    for pattern, key in zip(patterns, keys):
        pattern = re.compile(pattern)
        for line in stdout.splitlines():
            match = pattern.search(line)
            if match:
                for k, v in zip(key, map(float, match.groups())):
                    metric_dict[k] = v
    return metric_dict
