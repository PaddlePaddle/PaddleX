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

import tarfile
from pathlib import Path

from ..base import BaseExportor
from .model_list import MODELS


class TSADExportor(BaseExportor):
    """Image Classification Model Exportor"""

    entities = MODELS

    def get_config_path(self, weight_path):
        """
        get config path

        Args:
            weight_path (str): The path to the weight

        Returns:
            config_path (str): The path to the config

        """
        self.uncompress_tar_file()
        config_path = Path(self.export_config.weight_path).parent.parent / "config.yaml"
        return config_path

    def uncompress_tar_file(self):
        """unpackage the tar file containing training outputs and update weight path"""
        if tarfile.is_tarfile(self.export_config.weight_path):
            dest_path = Path(self.export_config.weight_path).parent
            with tarfile.open(self.export_config.weight_path, "r") as tar:
                tar.extractall(path=dest_path)
            self.export_config.weight_path = dest_path.joinpath(
                "best_accuracy.pdparams/best_model/model.pdparams"
            )
