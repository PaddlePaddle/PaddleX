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

from ....utils.cache import DEFAULT_CACHE_DIR
from ....utils.device import parse_device
from ....utils.download import download
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess


class SegModel(BaseModel):
    """Semantic Segmentation Model"""

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
        use_vdl: bool = True,
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
            use_vdl (bool, optional): enable VisualDL. Defaults to True.
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
            cli_args.append(CLIArgument("--iters", epochs_iters))

        # No need to handle `ips`

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        # For compatibility
        resume_dir = kwargs.pop("resume_dir", None)
        if resume_path is None and resume_dir is not None:
            resume_path = os.path.join(resume_dir, "model.pdparams")

        if resume_path is not None:
            # NOTE: We must use an absolute path here,
            # so we can run the scripts either inside or outside the repo dir.
            resume_path = abspath(resume_path)
            if os.path.basename(resume_path) != "model.pdparams":
                raise ValueError(f"{resume_path} has an incorrect file name.")
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"{resume_path} does not exist.")
            resume_dir = os.path.dirname(resume_path)
            opts_path = os.path.join(resume_dir, "model.pdopt")
            if not os.path.exists(opts_path):
                raise FileNotFoundError(f"{opts_path} must exist.")
            cli_args.append(CLIArgument("--resume_model", resume_dir))

        if dy2st:
            config.update_dy2st(dy2st)

        if use_vdl:
            cli_args.append(CLIArgument("--use_vdl"))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "train"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        save_interval = kwargs.pop("save_interval", None)
        if save_interval is not None:
            cli_args.append(CLIArgument("--save_interval", save_interval))

        do_eval = kwargs.pop("do_eval", True)
        repeats = kwargs.pop("repeats", None)
        seed = kwargs.pop("seed", None)

        profile = kwargs.pop("profile", None)
        if profile is not None:
            cli_args.append(CLIArgument("--profiler_options", profile))

        log_iters = kwargs.pop("log_iters", None)
        if log_iters is not None:
            cli_args.append(CLIArgument("--log_iters", log_iters))

        input_shape = kwargs.pop("input_shape", None)
        if input_shape is not None:
            cli_args.append(CLIArgument("--input_shape", *input_shape))

        # Benchmarking mode settings
        benchmark = kwargs.pop("benchmark", None)
        if benchmark is not None:
            envs = benchmark.get("env", None)
            seed = benchmark.get("seed", None)
            repeats = benchmark.get("repeats", None)
            do_eval = benchmark.get("do_eval", False)
            num_workers = benchmark.get("num_workers", None)
            config.update_log_ranks(device)
            amp = benchmark.get("amp", None)
            config.update_print_mem_info(benchmark.get("print_mem_info", True))
            config.update_shuffle(benchmark.get("shuffle", False))
            if repeats is not None:
                assert isinstance(repeats, int), "repeats must be an integer."
                cli_args.append(CLIArgument("--repeats", repeats))
            if num_workers is not None:
                assert isinstance(num_workers, int), "num_workers must be an integer."
                cli_args.append(CLIArgument("--num_workers", num_workers))
            if seed is not None:
                assert isinstance(seed, int), "seed must be an integer."
                cli_args.append(CLIArgument("--seed", seed))
            if amp in ["O1", "O2"]:
                cli_args.append(CLIArgument("--precision", "fp16"))
                cli_args.append(CLIArgument("--amp_level", amp))
            if envs is not None:
                for env_name, env_value in envs.items():
                    os.environ[env_name] = str(env_value)
        else:
            if amp is not None:
                if amp != "OFF":
                    cli_args.append(CLIArgument("--precision", "fp16"))
                    cli_args.append(CLIArgument("--amp_level", amp))
            if num_workers is not None:
                cli_args.append(CLIArgument("--num_workers", num_workers))
            if repeats is not None:
                cli_args.append(CLIArgument("--repeats", repeats))
            if seed is not None:
                cli_args.append(CLIArgument("--seed", seed))

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.set_val("uniform_output_enabled", uniform_output_enabled)
        config.set_val("pdx_model_name", self.name)
        if export_with_pir:
            config.set_val("export_with_pir", export_with_pir)

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)

            return self.runner.train(
                config_path, cli_args, device, ips, save_dir, do_eval=do_eval
            )

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
        cli_args.append(CLIArgument("--model_path", weight_path))

        if batch_size is not None:
            if batch_size != 1:
                raise ValueError("Batch size other than 1 is not supported.")

        # No need to handle `ips`

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        if amp is not None:
            if amp != "OFF":
                cli_args.append(CLIArgument("--precision", "fp16"))
                cli_args.append(CLIArgument("--amp_level", amp))

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
        cli_args.append(CLIArgument("--model_path", weight_path))

        input_path = abspath(input_path)
        cli_args.append(CLIArgument("--image_path", input_path))

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

    def analyse(self, weight_path, ips=None, device="gpu", save_dir=None, **kwargs):
        """analyse"""
        config = self.config.copy()
        cli_args = []

        weight_path = abspath(weight_path)
        cli_args.append(CLIArgument("--model_path", weight_path))

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "analysis"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            cp = self.runner.analyse(config_path, cli_args, device, ips)
            return cp

    def export(self, weight_path: str, save_dir: str, **kwargs) -> CompletedProcess:
        """export the dynamic model to static model

        Args:
            weight_path (str): the model weight file path that used to export.
            save_dir (str): the directory path to save export output.

        Returns:
            CompletedProcess: the result of exporting subprocess execution.
        """
        config = self.config.copy()
        cli_args = []

        if not weight_path.startswith("http"):
            weight_path = abspath(weight_path)
        else:
            filename = os.path.basename(weight_path)
            save_path = os.path.join(DEFAULT_CACHE_DIR, filename)
            download(weight_path, save_path, print_progress=True, overwrite=True)
            weight_path = save_path

        cli_args.append(CLIArgument("--model_path", weight_path))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "export"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        input_shape = kwargs.pop("input_shape", None)
        if input_shape is not None:
            cli_args.append(CLIArgument("--input_shape", *input_shape))

        try:
            output_op = config["output_op"]
        except:
            output_op = kwargs.pop("output_op", None)
        if output_op is not None:
            assert output_op in [
                "softmax",
                "argmax",
                "none",
            ], "`output_op` must be 'none', 'softmax' or 'argmax'."
            cli_args.append(CLIArgument("--output_op", output_op))

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.set_val("uniform_output_enabled", uniform_output_enabled)
        config.set_val("pdx_model_name", self.name)
        if export_with_pir:
            config.set_val("export_with_pir", export_with_pir)

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.export(config_path, cli_args, None)

    def infer(
        self,
        model_dir: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """predict image using infernece model

        Args:
            model_dir (str): the directory path of inference model files that would use to predict.
            input_path (str): the path of image that would be predict.
            device (str, optional): the running device. Defaults to 'gpu'.
            save_dir (str, optional): the directory path to save output. Defaults to None.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        config = self.config.copy()
        cli_args = []

        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        cli_args.append(CLIArgument("--image_path", input_path))

        if device is not None:
            device_type, _ = parse_device(device)
            cli_args.append(CLIArgument("--device", device_type))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "infer"))
        cli_args.append(CLIArgument("--save_dir", save_dir))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            deploy_config_path = os.path.join(model_dir, "inference.yml")
            return self.runner.infer(deploy_config_path, cli_args, device)

    def compression(
        self,
        weight_path: str,
        batch_size: int = None,
        learning_rate: float = None,
        epochs_iters: int = None,
        device: str = "gpu",
        use_vdl: bool = True,
        save_dir: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """compression model

        Args:
            weight_path (str): the path to weight file of model.
            batch_size (int, optional): the batch size value of compression training. Defaults to None.
            learning_rate (float, optional): the learning rate value of compression training. Defaults to None.
            epochs_iters (int, optional): the epochs or iters of compression training. Defaults to None.
            device (str, optional): the device to run compression training. Defaults to 'gpu'.
            use_vdl (bool, optional): whether or not to use VisualDL. Defaults to True.
            save_dir (str, optional): the directory to save output. Defaults to None.

        Returns:
            CompletedProcess: the result of compression subprocess execution.
        """
        # Update YAML config file
        # NOTE: In PaddleSeg, QAT does not use a different config file than regular training
        # Reusing `self.config` preserves the config items modified by the user when
        # `SegModel` is initialized with a `SegConfig` object.
        config = self.config.copy()
        train_cli_args = []
        export_cli_args = []

        weight_path = abspath(weight_path)
        train_cli_args.append(CLIArgument("--model_path", weight_path))

        if batch_size is not None:
            train_cli_args.append(CLIArgument("--batch_size", batch_size))

        if learning_rate is not None:
            train_cli_args.append(CLIArgument("--learning_rate", learning_rate))

        if epochs_iters is not None:
            train_cli_args.append(CLIArgument("--iters", epochs_iters))

        if device is not None:
            device_type, _ = parse_device(device)
            train_cli_args.append(CLIArgument("--device", device_type))

        if use_vdl:
            train_cli_args.append(CLIArgument("--use_vdl"))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "compress"))
        train_cli_args.append(CLIArgument("--save_dir", save_dir))
        # The exported model saved in a subdirectory named `export`
        export_cli_args.append(
            CLIArgument("--save_dir", os.path.join(save_dir, "export"))
        )

        input_shape = kwargs.pop("input_shape", None)
        if input_shape is not None:
            export_cli_args.append(CLIArgument("--input_shape", *input_shape))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.compression(
                config_path, train_cli_args, export_cli_args, device, save_dir
            )
