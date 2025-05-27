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


class FormulaRecRunner(BaseRunner):
    """Formula Recognition Runner"""

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
        if do_eval:
            # We simply pass here because in PaddleOCR periodic evaluation cannot be switched off
            pass
        else:
            inf = int(1.0e11)
            cmd.extend(["-o", f"Global.eval_batch_step={inf}"])
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
        cmd = [*args, "tools/eval.py", "-c", config_path]

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
        cmd = [self.python, "tools/infer_rec.py", "-c", config_path]
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
        cmd = [self.python, "tools/export_model.py", "-c", config_path]
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
        cmd = [self.python, "tools/infer/predict_rec.py", *cli_args]
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
        # Step 1: Train model
        args, env = self.distributed(device, log_dir=train_save_dir)
        cmd = [*args, "deploy/slim/quantization/quant.py", "-c", config_path]
        cp_train = self.run_cmd(
            cmd,
            env=env,
            switch_wdir=True,
            echo=True,
            silent=False,
            capture_output=True,
            log_path=self._get_train_log_path(train_save_dir),
        )

        # Step 2: Export model
        export_cli_args = [
            *export_cli_args,
            "-o",
            f"Global.checkpoints={train_save_dir}/latest",
        ]
        cmd = [
            self.python,
            "deploy/slim/quantization/export_model.py",
            "-c",
            config_path,
            *export_cli_args,
        ]
        cp_export = self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

        return cp_train, cp_export


def _extract_eval_metrics(stdout: str) -> dict:
    """extract evaluation metrics from training log

    Args:
        stdout (str): the training log

    Returns:
        dict: the training metric
    """
    import re

    def _lazy_split_lines(s):
        prev_idx = 0
        while True:
            curr_idx = s.find(os.linesep, prev_idx)
            if curr_idx == -1:
                curr_idx = len(s)
            yield s[prev_idx:curr_idx]
            prev_idx = curr_idx + len(os.linesep)
            if prev_idx >= len(s):
                break

    _DP = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    pattern_key_pairs = [
        (re.compile(r"acc:(_dp)$".replace("_dp", _DP)), "acc"),
        (re.compile(r"norm_edit_dis:(_dp)$".replace("_dp", _DP)), "norm_edit_dis"),
        (re.compile(r"Teacher_acc:(_dp)$".replace("_dp", _DP)), "teacher_acc"),
        (
            re.compile(r"Teacher_norm_edit_dis:(_dp)$".replace("_dp", _DP)),
            "teacher_norm_edit_dis",
        ),
        (re.compile(r"precision:(_dp)$".replace("_dp", _DP)), "precision"),
        (re.compile(r"recall:(_dp)$".replace("_dp", _DP)), "recall"),
        (re.compile(r"hmean:(_dp)$".replace("_dp", _DP)), "hmean"),
        (re.compile(r"exp_rate:(_dp)$".replace("_dp", _DP)), "exp_rate"),
    ]

    metric_dict = dict()
    start_match = False
    for line in _lazy_split_lines(stdout):
        if "metric eval" in line:
            start_match = True
        if start_match:
            for pattern, key in pattern_key_pairs:
                match = pattern.search(line)
                if match:
                    assert len(match.groups()) == 1
                    # Newer overwrites older
                    metric_dict[key] = float(match.group(1))
    return metric_dict
