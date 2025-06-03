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
from ....utils.device import parse_device
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess


class TextRecModel(BaseModel):
    """Text Recognition Model"""

    METRICS = [
        "acc",
        "norm_edit_dis",
        "Teacher_acc",
        "Teacher_norm_edit_dis",
        "precision",
        "recall",
        "hmean",
    ]

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

        if batch_size is not None:
            config.update_batch_size(batch_size)

        if learning_rate is not None:
            config.update_learning_rate(learning_rate)

        if epochs_iters is not None:
            config._update_epochs(epochs_iters)

        # No need to handle `ips`

        config.update_device(device)

        if resume_path is not None:
            resume_path = abspath(resume_path)
            config._update_checkpoints(resume_path)

        config._update_to_static(dy2st)

        config._update_amp(amp)

        if num_workers is not None:
            config.update_num_workers(num_workers, "train")

        config._update_use_vdl(use_vdl)

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            save_dir = abspath(config.get_train_save_dir())
        config._update_output_dir(save_dir)

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
            config.update_shuffle(benchmark.get("shuffle", False))
            config.update_cal_metrics(benchmark.get("cal_metrics", True))
            config.update_shared_memory(benchmark.get("shared_memory", True))
            config.update_print_mem_info(benchmark.get("print_mem_info", True))
            if num_workers is not None:
                config.update_num_workers(num_workers)
            if seed is not None:
                config.update_seed(seed)
            if envs is not None:
                for env_name, env_value in envs.items():
                    os.environ[env_name] = str(env_value)

        # PDX related settings
        device_type = device.split(":")[0]
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        config.update({"Global.uniform_output_enabled": uniform_output_enabled})
        config.update({"Global.model_name": self.name})
        config.update({"Global.export_with_pir": kwargs.pop("export_with_pir", False)})

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

        weight_path = abspath(weight_path)
        config._update_checkpoints(weight_path)

        if batch_size is not None:
            config.update_batch_size(batch_size)

        # No need to handle `ips`

        config.update_device(device)

        config._update_amp(amp)

        if num_workers is not None:
            config.update_num_workers(num_workers, "eval")

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            cp = self.runner.evaluate(config_path, [], device, ips)
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

        weight_path = abspath(weight_path)
        config.update_pretrained_weights(weight_path)

        input_path = abspath(input_path)
        config._update_infer_img(
            input_path, infer_list=kwargs.pop("input_list_path", None)
        )

        config.update_device(device)

        # TODO: Handle `device`
        logging.warning("`device` will not be used.")

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            save_dir = abspath(config.get_predict_save_dir())
        config._update_save_res_path(os.path.join(save_dir, "res.txt"))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.predict(config_path, [], device)

    def export(self, weight_path: str, save_dir: str, **kwargs) -> CompletedProcess:
        """export the dynamic model to static model

        Args:
            weight_path (str): the model weight file path that used to export.
            save_dir (str): the directory path to save export output.

        Returns:
            CompletedProcess: the result of exporting subprocess execution.
        """
        config = self.config.copy()

        device = kwargs.pop("device", None)
        if device:
            config.update_device(device)

        if not weight_path.startswith("http"):
            weight_path = abspath(weight_path)
        config.update_pretrained_weights(weight_path)

        save_dir = abspath(save_dir)
        config._update_save_inference_dir(save_dir)

        class_path = kwargs.pop("class_path", None)
        if class_path is not None:
            config.update_class_path(class_path)

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        config.update({"Global.uniform_output_enabled": uniform_output_enabled})
        config.update({"Global.model_name": self.name})
        config.update({"Global.export_with_pir": kwargs.pop("export_with_pir", False)})

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.export(config_path, [], None, save_dir)

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
        cli_args.append(CLIArgument("--rec_model_dir", model_dir))

        input_path = abspath(input_path)
        cli_args.append(CLIArgument("--image_dir", input_path))

        device_type, _ = parse_device(device)
        cli_args.append(CLIArgument("--use_gpu", str(device_type == "gpu")))

        if save_dir is not None:
            logging.warning("`save_dir` will not be used.")

        dict_path = kwargs.pop("dict_path", None)
        if dict_path is not None:
            dict_path = abspath(dict_path)
        else:
            dict_path = config.get_label_dict_path()
        cli_args.append(CLIArgument("--rec_char_dict_path", dict_path))

        model_type = config._get_model_type()
        cli_args.append(CLIArgument("--rec_algorithm", model_type))
        infer_shape = config._get_infer_shape()
        if infer_shape is not None:
            cli_args.append(CLIArgument("--rec_image_shape", infer_shape))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.infer(config_path, cli_args, device)

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
        config = self.config.copy()
        export_cli_args = []

        weight_path = abspath(weight_path)
        config.update_pretrained_weights(weight_path)

        if batch_size is not None:
            config.update_batch_size(batch_size)

        if learning_rate is not None:
            config.update_learning_rate(learning_rate)

        if epochs_iters is not None:
            config._update_epochs(epochs_iters)

        config.update_device(device)

        config._update_use_vdl(use_vdl)

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            save_dir = abspath(config.get_train_save_dir())
        config._update_output_dir(save_dir)
        export_cli_args.append(
            CLIArgument(
                "-o", f"Global.save_inference_dir={os.path.join(save_dir, 'export')}"
            )
        )

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)

            return self.runner.compression(
                config_path, [], export_cli_args, device, save_dir
            )
