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
from ...base.utils.arg import CLIArgument
from ...base.utils.subprocess import CompletedProcess
from ..text_rec.model import TextRecModel


class TableRecModel(TextRecModel):
    """Table Recognition Model"""

    METRICS = ["acc"]

    def predict(
        self,
        weight_path: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs
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
        config._update_infer_img(input_path)

        # TODO: Handle `device`
        logging.warning("`device` will not be used.")

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            save_dir = abspath(config.get_predict_save_dir())
        config._update_save_res_path(save_dir)

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.predict(config_path, [], device)

    def infer(
        self,
        model_dir: str,
        input_path: str,
        device: str = "gpu",
        save_dir: str = None,
        **kwargs
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
        cli_args.append(CLIArgument("--table_model_dir", model_dir))

        input_path = abspath(input_path)
        cli_args.append(CLIArgument("--image_dir", input_path))

        device_type, _ = parse_device(device)
        cli_args.append(CLIArgument("--use_gpu", str(device_type == "gpu")))

        if save_dir is not None:
            save_dir = abspath(save_dir)
        else:
            # `save_dir` is None
            save_dir = abspath(os.path.join("output", "infer"))
        cli_args.append(CLIArgument("--output", save_dir))

        dict_path = kwargs.pop("dict_path", None)
        if dict_path is not None:
            dict_path = abspath(dict_path)
        else:
            dict_path = config.get_label_dict_path()
        cli_args.append(CLIArgument("--table_char_dict_path", dict_path))

        model_type = config._get_model_type()
        cli_args.append(CLIArgument("--table_algorithm", model_type))
        infer_shape = config._get_infer_shape()
        if infer_shape is not None:
            cli_args.append(CLIArgument("--table_max_len", infer_shape))

        self._assert_empty_kwargs(kwargs)

        with self._create_new_config_file() as config_path:
            config.dump(config_path)
            return self.runner.infer(config_path, cli_args, device)
