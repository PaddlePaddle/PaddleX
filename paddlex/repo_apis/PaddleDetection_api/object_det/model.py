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

import json
import os

from ....utils import logging
from ....utils.device import parse_device
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess
from .config import DetConfig
from .official_categories import official_categories


class DetModel(BaseModel):
    """Object Detection Model"""

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
            config.update_batch_size(batch_size, "train")
        if learning_rate is not None:
            config.update_learning_rate(learning_rate)
        if epochs_iters is not None:
            config.update_epochs(epochs_iters)
            config.update_cossch_epoch(epochs_iters)
        device_type, _ = parse_device(device)
        config.update_device(device_type)
        if resume_path is not None:
            assert resume_path.endswith(
                ".pdparams"
            ), "resume_path should be endswith .pdparam"
            resume_dir = resume_path[0:-9]
            cli_args.append(CLIArgument("--resume", resume_dir))
        if dy2st:
            cli_args.append(CLIArgument("--to_static"))
        if num_workers is not None:
            config.update_num_workers(num_workers)
        if save_dir is None:
            save_dir = abspath(config.get_train_save_dir())
        else:
            save_dir = abspath(save_dir)
        config.update_save_dir(save_dir)
        if use_vdl:
            cli_args.append(CLIArgument("--use_vdl", use_vdl))
            cli_args.append(CLIArgument("--vdl_log_dir", save_dir))

        do_eval = kwargs.pop("do_eval", True)
        enable_ce = kwargs.pop("enable_ce", None)

        profile = kwargs.pop("profile", None)
        if profile is not None:
            cli_args.append(CLIArgument("--profiler_options", profile))

        # Benchmarking mode settings
        benchmark = kwargs.pop("benchmark", None)
        if benchmark is not None:
            envs = benchmark.get("env", None)
            amp = benchmark.get("amp", None)
            do_eval = benchmark.get("do_eval", False)
            num_workers = benchmark.get("num_workers", None)
            config.update_log_ranks(device)
            config.update_shuffle(benchmark.get("shuffle", False))
            config.update_shared_memory(benchmark.get("shared_memory", True))
            config.update_print_mem_info(benchmark.get("print_mem_info", True))
            if num_workers is not None:
                config.update_num_workers(num_workers)
            if amp == "O1":
                # TODO: ppdet only support ampO1
                cli_args.append(CLIArgument("--amp"))
            if envs is not None:
                for env_name, env_value in envs.items():
                    os.environ[env_name] = str(env_value)
            # set seed to 0 for benchmark mode by enable_ce
            cli_args.append(CLIArgument("--enable_ce", True))
        else:
            if amp != "OFF" and amp is not None:
                # TODO: consider amp is O1 or O2 in ppdet
                cli_args.append(CLIArgument("--amp"))
            if enable_ce is not None:
                cli_args.append(CLIArgument("--enable_ce", enable_ce))

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
            return self.runner.train(
                config_path, cli_args, device, ips, save_dir, do_eval=do_eval
            )

    def evaluate(
        self,
        weight_path: str,
        batch_size: int = None,
        ips: bool = None,
        device: bool = "gpu",
        amp: bool = "OFF",
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
        config.update_weights(weight_path)
        if batch_size is not None:
            config.update_batch_size(batch_size, "eval")
        device_type, device_ids = parse_device(device)
        if device_ids is not None and len(device_ids) > 1:
            raise ValueError(
                f"multi-{device_type} evaluation is not supported. Please use a single {device_type}."
            )
        config.update_device(device_type)
        if amp != "OFF":
            # TODO: consider amp is O1 or O2 in ppdet
            cli_args.append(CLIArgument("--amp"))
        if num_workers is not None:
            config.update_num_workers(num_workers)

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            cp = self.runner.evaluate(config_path, cli_args, device, ips)
            return cp

    def predict(
        self,
        input_path: str,
        weight_path: str,
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

        input_path = abspath(input_path)
        if os.path.isfile(input_path):
            cli_args.append(CLIArgument("--infer_img", input_path))
        else:
            cli_args.append(CLIArgument("--infer_dir", input_path))
        if "infer_list" in kwargs:
            infer_list = abspath(kwargs.get("infer_list"))
            cli_args.append(CLIArgument("--infer_list", infer_list))
        if "visualize" in kwargs:
            cli_args.append(CLIArgument("--visualize", kwargs["visualize"]))
        if "save_results" in kwargs:
            cli_args.append(CLIArgument("--save_results", kwargs["save_results"]))
        if "save_threshold" in kwargs:
            cli_args.append(CLIArgument("--save_threshold", kwargs["save_threshold"]))
        if "rtn_im_file" in kwargs:
            cli_args.append(CLIArgument("--rtn_im_file", kwargs["rtn_im_file"]))
        weight_path = abspath(weight_path)
        config.update_weights(weight_path)
        device_type, _ = parse_device(device)
        config.update_device(device_type)
        if save_dir is not None:
            save_dir = abspath(save_dir)
            cli_args.append(CLIArgument("--output_dir", save_dir))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.predict(config_path, cli_args, device)

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

        device = kwargs.pop("device", None)
        if device:
            device_type, _ = parse_device(device)
            config.update_device(device_type)

        if not weight_path.startswith("http"):
            weight_path = abspath(weight_path)
        config.update_weights(weight_path)
        save_dir = abspath(save_dir)
        cli_args.append(CLIArgument("--output_dir", save_dir))
        input_shape = kwargs.pop("input_shape", None)
        if input_shape is not None:
            cli_args.append(
                CLIArgument("-o", f"TestReader.inputs_def.image_shape={input_shape}")
            )

        use_trt = kwargs.pop("use_trt", None)
        if use_trt is not None:
            cli_args.append(CLIArgument("-o", f"trt={bool(use_trt)}"))

        exclude_nms = kwargs.pop("exclude_nms", None)
        if exclude_nms is not None:
            cli_args.append(CLIArgument("-o", f"exclude_nms={bool(exclude_nms)}"))

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.update({"uniform_output_enabled": uniform_output_enabled})
        config.update({"pdx_model_name": self.name})
        if export_with_pir:
            config.update({"export_with_pir": export_with_pir})

        if self.name in official_categories.keys():
            anno_val_file = abspath(
                os.path.join(
                    config.TestDataset["dataset_dir"], config.TestDataset["anno_path"]
                )
            )
            if anno_val_file == None or (not os.path.isfile(anno_val_file)):
                categories = official_categories[self.name]
                temp_anno = {"images": [], "annotations": [], "categories": categories}
                with self._create_new_val_json_file() as anno_file:
                    json.dump(temp_anno, open(anno_file, "w"))
                    config.update(
                        {"TestDataset": {"dataset_dir": "", "anno_path": anno_file}}
                    )
                    logging.warning(
                        f"{self.name} does not have validate annotations, use {anno_file} default instead."
                    )
                    self._assert_empty_kwargs(kwargs)
                    with self._create_new_config_file() as config_path:
                        config.dump(config_path)
                        return self.runner.export(config_path, cli_args, None)

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
    ):
        """predict image using infernece model

        Args:
            model_dir (str): the directory path of inference model files that would use to predict.
            input_path (str): the path of image that would be predict.
            device (str, optional): the running device. Defaults to 'gpu'.
            save_dir (str, optional): the directory path to save output. Defaults to None.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        cli_args = []
        cli_args.append(CLIArgument("--model_dir", model_dir))
        cli_args.append(CLIArgument("--image_file", input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument("--output_dir", save_dir))
        device_type, _ = parse_device(device)
        cli_args.append(CLIArgument("--device", device_type))

        self._assert_empty_kwargs(kwargs)

        return self.runner.infer(cli_args, device)

    def compression(
        self,
        weight_path: str,
        batch_size: int = None,
        learning_rate: float = None,
        epochs_iters: int = None,
        device: str = None,
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
        weight_path = abspath(weight_path)
        if save_dir is None:
            save_dir = self.config["save_dir"]
        save_dir = abspath(save_dir)

        config = self.config.copy()
        cps_config = DetConfig(
            self.name, config_path=self.model_info["auto_compression_config_path"]
        )
        train_cli_args = []
        export_cli_args = []

        cps_config.update_pretrained_weights(weight_path)
        if batch_size is not None:
            cps_config.update_batch_size(batch_size, "train")
        if learning_rate is not None:
            cps_config.update_learning_rate(learning_rate)
        if epochs_iters is not None:
            cps_config.update_epochs(epochs_iters)
        if device is not None:
            device_type, _ = parse_device(device)
            config.update_device(device_type)
        if save_dir is not None:
            save_dir = abspath(config.get_train_save_dir())
        else:
            save_dir = abspath(save_dir)
        cps_config.update_save_dir(save_dir)
        if use_vdl:
            train_cli_args.append(CLIArgument("--use_vdl", use_vdl))
            train_cli_args.append(CLIArgument("--vdl_log_dir", save_dir))

        export_cli_args.append(
            CLIArgument("--output_dir", os.path.join(save_dir, "export"))
        )

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            # TODO: refactor me
            cps_config_path = config_path[0:-4] + "_compression" + config_path[-4:]
            cps_config.dump(cps_config_path)
            train_cli_args.append(CLIArgument("--slim_config", cps_config_path))
            export_cli_args.append(CLIArgument("--slim_config", cps_config_path))

            self._assert_empty_kwargs(kwargs)

            self.runner.compression(
                config_path, train_cli_args, export_cli_args, device, save_dir
            )
