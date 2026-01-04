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

from ....utils import logging
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess


class VideoClsModel(BaseModel):
    """Video Classification Model"""

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
        if resume_path is not None:
            resume_path = abspath(resume_path)

        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config.update_device(device)
            config._update_to_static(dy2st)
            config._update_use_vdl(use_vdl)

            if batch_size is not None:
                config.update_batch_size(batch_size)
            if learning_rate is not None:
                config.update_learning_rate(learning_rate)
            if epochs_iters is not None:
                config._update_epochs(epochs_iters)
            config._update_checkpoints(resume_path)
            if save_dir is not None:
                save_dir = abspath(save_dir)
            else:
                # `save_dir` is None
                save_dir = abspath(config.get_train_save_dir())
            config._update_output_dir(save_dir)
            if num_workers is not None:
                config.update_num_workers(num_workers)

            cli_args = []
            do_eval = kwargs.pop("do_eval", True)
            profile = kwargs.pop("profile", None)
            if profile is not None:
                cli_args.append(CLIArgument("--profiler_options", profile))

            # Benchmarking mode settings
            benchmark = kwargs.pop("benchmark", None)
            if benchmark is not None:
                envs = benchmark.get("env", None)
                seed = benchmark.get("seed", None)
                do_eval = benchmark.get("do_eval", False)
                num_workers = benchmark.get("num_workers", None)
                config.update_log_ranks(device)
                config._update_amp(benchmark.get("amp", None))
                config.update_dali(benchmark.get("dali", False))
                config.update_shuffle(benchmark.get("shuffle", False))
                config.update_shared_memory(benchmark.get("shared_memory", True))
                config.update_print_mem_info(benchmark.get("print_mem_info", True))
                if num_workers is not None:
                    config.update_num_workers(num_workers)
                if seed is not None:
                    config.update_seed(seed)
                if envs is not None:
                    for env_name, env_value in envs.items():
                        os.environ[env_name] = str(env_value)
            else:
                config._update_amp(amp)
            # PDX related settings
            device_type = device.split(":")[0]
            uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
            config.update({"Global.uniform_output_enabled": uniform_output_enabled})
            config.update({"Global.pdx_model_name": self.name})

            config.dump(config_path)
            self._assert_empty_kwargs(kwargs)
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

        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config._update_amp(amp)
            config.update_device(device)
            config.update_pretrained_weights(weight_path)
            if batch_size is not None:
                config.update_batch_size(batch_size)
            if num_workers is not None:
                config.update_num_workers(num_workers)

            config.dump(config_path)

            self._assert_empty_kwargs(kwargs)

            cp = self.runner.evaluate(config_path, [], device, ips)
            return cp

    def predict(
        self,
        weight_path: str,
        input_path: str,
        input_list_path: str = None,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """predict using specified weight

        Args:
            weight_path (str): the path of model weight file used to predict.
            input_path (str): the path of image file to be predicted.
            input_list_path (str, optional): the paths of images to be predicted if is not None. Defaults to None.
            device (str, optional): the running device. Defaults to 'gpu'.
            save_dir (str, optional): the directory path to save predict output. Defaults to None.

        Returns:
            CompletedProcess: the result of predicting subprocess execution.
        """
        input_path = abspath(input_path)
        if input_list_path:
            input_list_path = abspath(input_list_path)

        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config.update_pretrained_weights(weight_path)
            config._update_predict_img(input_path, input_list_path)
            config.update_device(device)
            config._update_save_predict_result(save_dir)

            config.dump(config_path)

            self._assert_empty_kwargs(kwargs)

            return self.runner.predict(config_path, [], device)

    def export(self, weight_path: str, save_dir: str, **kwargs) -> CompletedProcess:
        """export the dynamic model to static model

        Args:
            weight_path (str): the model weight file path that used to export.
            save_dir (str): the directory path to save export output.

        Returns:
            CompletedProcess: the result of exporting subprocess execution.
        """
        if not weight_path.startswith(("http://", "https://")):
            weight_path = abspath(weight_path)
        save_dir = abspath(save_dir)

        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config.update_pretrained_weights(weight_path)
            config._update_save_inference_dir(save_dir)
            device = kwargs.pop("device", None)
            if device:
                config.update_device(device)
            # PDX related settings
            uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
            config.update({"Global.uniform_output_enabled": uniform_output_enabled})
            config.update({"Global.pdx_model_name": self.name})

            config.dump(config_path)

            self._assert_empty_kwargs(kwargs)

            return self.runner.export(config_path, [], None, save_dir)

    def infer(
        self,
        model_dir: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        dict_path: str = None,
        **kwargs,
    ) -> CompletedProcess:
        """predict image using infernece model

        Args:
            model_dir (str): the directory path of inference model files that would use to predict.
            input_path (str): the path of image that would be predict.
            device (str, optional): the running device. Defaults to 'gpu'.
            save_dir (str, optional): the directory path to save output. Defaults to None.
            dict_path (str, optional): the label dict file path. Defaults to None.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        if save_dir is not None:
            logging.warning("`save_dir` will not be used.")
        config_path = os.path.join(model_dir, "inference.yml")
        config = self.config.copy()
        config.load(config_path)
        config._update_inference_model_dir(model_dir)
        config._update_infer_img(input_path)
        config._update_infer_device(device)
        if dict_path is not None:
            dict_path = abspath(dict_path)
            config.update_label_dict_path(dict_path)
        if "enable_mkldnn" in kwargs:
            config._update_enable_mkldnn(kwargs.pop("enable_mkldnn"))

        with self._create_new_config_file() as config_path:
            config.dump(config_path)

            self._assert_empty_kwargs(kwargs)

            return self.runner.infer(config_path, [], device)

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

        with self._create_new_config_file() as config_path:
            # Update YAML config file
            config = self.config.copy()
            config._update_amp(None)
            config.update_device(device)
            config._update_use_vdl(use_vdl)
            config._update_slim_config(self.model_info["auto_compression_config_path"])
            config.update_pretrained_weights(weight_path)

            if batch_size is not None:
                config.update_batch_size(batch_size)
            if learning_rate is not None:
                config.update_learning_rate(learning_rate)
            if epochs_iters is not None:
                config._update_epochs(epochs_iters)
            if save_dir is not None:
                save_dir = abspath(save_dir)
            else:
                # `save_dir` is None
                save_dir = abspath(config.get_train_save_dir())
            config._update_output_dir(save_dir)
            config.dump(config_path)

            export_cli_args = []
            export_cli_args.append(
                CLIArgument(
                    "-o",
                    f"Global.save_inference_dir={os.path.join(save_dir, 'export')}",
                )
            )

            self._assert_empty_kwargs(kwargs)

            return self.runner.compression(
                config_path, [], export_cli_args, device, save_dir
            )
