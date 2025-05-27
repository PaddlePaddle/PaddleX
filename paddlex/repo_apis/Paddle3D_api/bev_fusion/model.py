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


import os.path as osp

from ....utils import logging
from ....utils.misc import abspath
from ...base import BaseModel
from ...base.utils.arg import CLIArgument
from .runner import raise_unsupported_api_error


class BEVFusionModel(BaseModel):

    def train(
        self,
        batch_size=None,
        learning_rate=None,
        epochs_iters=None,
        pretrained=None,
        ips=None,
        device="gpu",
        resume_path=None,
        dy2st=False,
        amp="OFF",
        num_workers=None,
        use_vdl=True,
        save_dir=None,
        **kwargs,
    ):
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if not use_vdl:
            logging.warning("Currently, VisualDL cannot be disabled during training.")
        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(osp.join("output", "train"))

        if dy2st:
            raise ValueError(f"`dy2st`={dy2st} is not supported.")
        if device in ("cpu", "gpu"):
            logging.warning(
                f"The device type to use will be automatically determined, which may differ from the specified type: {repr(device)}."
            )

        # Update YAML config file
        config = self.config.copy()
        if epochs_iters is not None:
            config.update_iters(epochs_iters)
        if amp is not None:
            if amp != "OFF":
                config._update_amp(amp)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument("--batch_size", batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument("--learning_rate", learning_rate))
        if num_workers is not None:
            cli_args.append(CLIArgument("--num_workers", num_workers))
        if resume_path is not None:
            if save_dir is not None:
                raise ValueError(
                    "When `resume_path` is not None, `save_dir` must be set to None."
                )
            model_dir = osp.dirname(resume_path)
            cli_args.append(CLIArgument("--resume"))
            cli_args.append(CLIArgument("--save_dir", model_dir))
        if save_dir is not None:
            cli_args.append(CLIArgument("--save_dir", save_dir))
        if pretrained is not None:
            cli_args.append(CLIArgument("--model", abspath(pretrained)))

        do_eval = kwargs.pop("do_eval", True)

        profile = kwargs.pop("profile", None)
        if profile is not None:
            cli_args.append(CLIArgument("--profiler_options", profile))

        log_interval = kwargs.pop("log_interval", 1)
        if log_interval is not None:
            cli_args.append(CLIArgument("--log_interval", log_interval))

        save_interval = kwargs.pop("save_interval", 1)
        if save_interval is not None:
            cli_args.append(CLIArgument("--save_interval", save_interval))

        seed = kwargs.pop("seed", None)
        if seed is not None:
            cli_args.append(CLIArgument("--seed", seed))

        self._assert_empty_kwargs(kwargs)

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.update({"uniform_output_enabled": uniform_output_enabled})
        config.update({"pdx_model_name": self.name})
        if export_with_pir:
            config.update({"export_with_pir": export_with_pir})

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.train(
                config_path, cli_args, device, ips, save_dir, do_eval=do_eval
            )

    def evaluate(
        self,
        weight_path,
        batch_size=None,
        ips=None,
        device="gpu",
        amp="OFF",
        num_workers=None,
        **kwargs,
    ):
        weight_path = abspath(weight_path)

        if device in ("cpu", "gpu"):
            logging.warning(
                f"The device type to use will be automatically determined, which may differ from the specified type: {repr(device)}."
            )

        # Update YAML config file
        config = self.config.copy()

        if amp is not None:
            if amp != "OFF":
                raise ValueError("AMP evaluation is not supported.")

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument("--model", weight_path))
        if batch_size is not None:
            cli_args.append(CLIArgument("--batch_size", batch_size))
            if batch_size != 1:
                raise ValueError("Batch size other than 1 is not supported.")
        if num_workers is not None:
            cli_args.append(CLIArgument("--num_workers", num_workers))

        self._assert_empty_kwargs(kwargs)

        # PDX related settings
        uniform_output_enabled = kwargs.pop("uniform_output_enabled", True)
        export_with_pir = kwargs.pop("export_with_pir", False)
        config.update({"uniform_output_enabled": uniform_output_enabled})
        config.update({"pdx_model_name": self.name})
        if export_with_pir:
            config.update({"export_with_pir": export_with_pir})

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            cp = self.runner.evaluate(config_path, cli_args, device, ips)
            return cp

    def predict(self, weight_path, input_path, device="gpu", save_dir=None, **kwargs):
        raise_unsupported_api_error("predict", self.__class__)

    def export(self, weight_path, save_dir, **kwargs):
        if not weight_path.startswith("http"):
            weight_path = abspath(weight_path)
        save_dir = abspath(save_dir)

        # Update YAML config file
        config = self.config.copy()

        # Parse CLI arguments
        cli_args = []
        if weight_path is not None:
            cli_args.append(CLIArgument("--model", weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument("--save_dir", save_dir))

        cli_args.append(CLIArgument("--save_name", "inference"))
        cli_args.append(CLIArgument("--save_inference_yml"))

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
            return self.runner.export(config_path, cli_args, None)

    def infer(self, model_dir, device="gpu", **kwargs):
        model_dir = abspath(model_dir)

        # Parse CLI arguments
        cli_args = []
        model_file_path = osp.join(model_dir, ".pdmodel")
        params_file_path = osp.join(model_dir, ".pdiparams")
        cli_args.append(CLIArgument("--model_file", model_file_path))
        cli_args.append(CLIArgument("--params_file", params_file_path))
        if device is not None:
            device_type, _ = self.runner.parse_device(device)
            if device_type not in ("cpu", "gpu"):
                raise ValueError(f"`device`={repr(device)} is not supported.")
        infer_dir = osp.join(self.runner.runner_root_path, self.model_info["infer_dir"])
        self._assert_empty_kwargs(kwargs)
        # The inference script does not require a config file
        return self.runner.infer(None, cli_args, device, infer_dir, None)

    def compression(
        self,
        weight_path,
        ann_file=None,
        class_names=None,
        batch_size=None,
        learning_rate=None,
        epochs_iters=None,
        device="gpu",
        use_vdl=True,
        save_dir=None,
        **kwargs,
    ):
        raise_unsupported_api_error("compression", self.__class__)
