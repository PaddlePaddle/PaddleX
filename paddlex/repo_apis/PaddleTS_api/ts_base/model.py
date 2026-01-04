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

from ....utils.device import parse_device
from ....utils.errors import raise_unsupported_api_error
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess


class TSModel(BaseModel):
    """TS Model"""

    def train(
        self,
        batch_size: int = None,
        learning_rate: float = None,
        epochs_iters: int = None,
        ips: str = None,
        device: str = "gpu",
        resume_path: str = None,
        dy2st: bool = False,
        amp: str = "OFF",
        num_workers: int = None,
        use_vdl: bool = False,
        save_dir: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """train self

        Args:
            batch_size (int, optional): the train batch size value. Defaults to None.
            learning_rate (float, optional): the train learning rate value. Defaults to None.
            epochs_iters (int, optional): the train epochs value. Defaults to None.
            ips (str, optional): the ip addresses of nodes when using distribution. Defaults to None.
            device (str, optional): the running device. Defaults to 'gpu'.
            resume_path (str, optional): the checkpoint file path to resume training. Train from scratch if it is set
                to None. Defaults to None.
            dy2st (bool, optional): Enable dynamic to static. Defaults to False.
            amp (str, optional): the amp settings. Defaults to 'OFF'.
            num_workers (int, optional): the workers number. Defaults to None.
            use_vdl (bool, optional): enable VisualDL. Defaults to False.
            save_dir (str, optional): the directory path to save train output. Defaults to None.

        Returns:
           CompletedProcess: the result of training subprocess execution.
        """
        config = self.config.copy()
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument("--batch_size", batch_size))

        if learning_rate is not None:
            cli_args.append(CLIArgument("--learning_rate", learning_rate))

        if epochs_iters is not None:
            cli_args.append(CLIArgument("--epoch", epochs_iters))

        if resume_path:
            raise ValueError("`resume_path` is not supported.")
        # No need to handle `ips`
        benchmark = kwargs.pop("benchmark", None)
        if benchmark is not None:
            amp = benchmark.get("amp", None)
            if amp in ["O1", "O2"]:
                config.update_amp(amp)

        if use_vdl:
            raise ValueError(f"`use_vdl`={use_vdl} is not supported.")

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "train"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        # Benchmarking mode settings
        benchmark = kwargs.pop("benchmark", None)
        if benchmark is not None:
            envs = benchmark.get("env", None)
            num_workers = benchmark.get("num_workers", None)
            config.update_log_ranks(device)
            config.update_print_mem_info(benchmark.get("print_mem_info", True))
            if num_workers is not None:
                assert isinstance(num_workers, int), "num_workers must be an integer"
                cli_args.append(CLIArgument("--num_workers", num_workers))
            if envs is not None:
                for env_name, env_value in envs.items():
                    os.environ[env_name] = str(env_value)
        else:
            if num_workers is not None:
                cli_args.append(CLIArgument("--num_workers", num_workers))
        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.update({"uniform_output_enabled": uniform_output_enabled})
        config.update({"pdx_model_name": self.name})
        if export_with_pir:
            config.update({"export_with_pir": export_with_pir})

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.train(config_path, cli_args, device, ips, save_dir)

    def evaluate(
        self,
        weight_path: str,
        batch_size: int = None,
        ips: str = None,
        device: str = "gpu",
        amp: str = "OFF",
        num_workers: int = None,
        **kwargs,
    ) -> CompletedProcess:
        """evaluate self using specified weight

        Args:
            weight_path (str): the path of model weight file to be evaluated.
            batch_size (int, optional): the batch size value in evaluating. Defaults to None.
            ips (str, optional): the ip addresses of nodes when using distribution. Defaults to None.
            device (str, optional): the running device. Defaults to 'gpu'.
            amp (str, optional): the AMP setting. Defaults to 'OFF'.
            num_workers (int, optional): the workers number in evaluating. Defaults to None.

        Returns:
            CompletedProcess: the result of evaluating subprocess execution.
        """
        config = self.config.copy()
        cli_args = []

        weight_path = abspath(weight_path)
        cli_args.append(CLIArgument("--checkpoints", weight_path))

        if batch_size is not None:
            if batch_size != 1:
                raise ValueError("Batch size other than 1 is not supported.")

        # No need to handle `ips`
        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        if amp is not None:
            if amp != "OFF":
                raise ValueError(f"`amp`={amp} is not supported.")

        if num_workers is not None:
            cli_args.append(CLIArgument("--num_workers", num_workers))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            cp = self.runner.evaluate(config_path, cli_args, device, ips)
            return cp

    def predict(
        self,
        weight_path: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """predict using specified weight

        Args:
            weight_path (str): the path of model weight file used to predict.
            input_path (str): the path of image file to be predicted.
            device (str, optional): the running device. Defaults to 'gpu'.
            save_dir (str, optional): the directory path to save predict output. Defaults to None.

        Returns:
            CompletedProcess: the result of predicting subprocess execution.
        """
        config = self.config.copy()
        cli_args = []

        weight_path = abspath(weight_path)
        cli_args.append(CLIArgument("--checkpoints", weight_path))

        input_path = abspath(input_path)
        cli_args.append(CLIArgument("--csv_path", input_path))

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "predict"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.predict(config_path, cli_args, device)

    def export(
        self, weight_path: str, save_dir: str = None, device: str = "gpu", **kwargs
    ):
        """export"""
        if not weight_path.startswith(("http://", "https://")):
            weight_path = abspath(weight_path)
        save_dir = abspath(save_dir)
        cli_args = []

        cli_args.append(CLIArgument("--checkpoints", weight_path))
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            save_dir = abspath(os.path.join("output", "inference"))
        cli_args.append(CLIArgument("--save_dir", save_dir))
        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))
        export_with_pir = kwargs.pop("export_with_pir", False)
        self._assert_empty_kwargs(kwargs)
        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config.update_pretrained_weights(weight_path)
            config.update({"pdx_model_name": self.name})
            if export_with_pir:
                config.update({"export_with_pir": export_with_pir})
            config.dump(config_path)

            return self.runner.export(config_path, cli_args, device)

    def infer(
        self,
        model_dir: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs,
    ):
        """infer"""
        raise_unsupported_api_error("infer", self.__class__)

    def compression(
        self,
        weight_path: str,
        batch_size=None,
        learning_rate=None,
        epochs_iters=None,
        device: str = "gpu",
        use_vdl=True,
        save_dir=None,
        **kwargs,
    ):
        """compression"""
        raise_unsupported_api_error("compression", self.__class__)
