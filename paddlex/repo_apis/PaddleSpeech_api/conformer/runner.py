# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
from paddlespeech.cli import ASRTrainingExecutor
from ...base import BaseRunner


def raise_unsupported_api_error(api_name, cls=None):
    # TODO: Automatically extract `api_name` and `cls` from stack frame
    if cls is not None:
        name = f"{cls.__name__}.{api_name}"
    else:
        name = api_name
    raise UnsupportedAPIError(f"The API `{name}` is not supported.")


class UnsupportedAPIError(Exception):
    pass


class ChunkConformerRunner(BaseRunner):
    def train(self, config_path, cli_args, device, ips, save_dir, do_eval=True):
        args, env = self.distributed(device, ips, log_dir=save_dir)
        cmd = [*args, "tools/train.py"]
        if do_eval:
            cmd.append("--do_eval")
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

    def evaluate(self, config_path, cli_args, device, ips):
        args, env = self.distributed(device, ips)
        cmd = [*args, "tools/evaluate.py", "--config", config_path, *cli_args]
        cp = self.run_cmd(
            cmd, env=env, switch_wdir=True, echo=True, silent=False, capture_output=True
        )
        if cp.returncode == 0:
            metric_dict = _extract_eval_metrics(cp.stdout)
            cp.metrics = metric_dict
        return cp

    def predict(self, config_path, cli_args, device):
        raise_unsupported_api_error("predict", self.__class__)

    def export(self, config_path, cli_args, device):
        # `device` unused
        cmd = [self.python, "tools/export.py", "--config", config_path, *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device, infer_dir, save_dir=None):
        # `config_path` and `device` unused
        cmd = [self.python, "infer.py", *cli_args]
        python_infer_dir = os.path.join(infer_dir, "python")
        cp = self.run_cmd(cmd, switch_wdir=python_infer_dir, echo=True, silent=False)
        return cp

    def compression(
        self, config_path, train_cli_args, export_cli_args, device, train_save_dir
    ):
        raise_unsupported_api_error("compression", self.__class__)

    def create_dataloader(self, dataset_config):
        """Create data loader with chunked audio processing"""
        return ASRTrainingExecutor.get_dataloader(
            dataset_config.type,
            data_dir=dataset_config.data_dir,
            sample_rate=dataset_config.sample_rate,
            lang=dataset_config.lang,
            batch_size=self.config.batch_size,
            chunk_size=self.model.chunk_size,
        )


def _extract_eval_metrics(stdout):
    import re

    _DP = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    metrics = ["CER", "WER"]
    patterns = {}
    for metric in metrics:
        pattern = f"{metric}: (_dp)".replace("_dp", _DP)
        patterns[metric] = pattern

    metric_dict = dict()

    # TODO: Use lazy version to make it more efficient
    lines = stdout.splitlines()
    for line in lines:
        for m in patterns:
            p = re.compile(patterns[m])
            match = p.search(line)
            if match:
                metric_dict[m] = float(match.groups()[0])

    return metric_dict
