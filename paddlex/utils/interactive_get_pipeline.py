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

import shutil
from pathlib import Path

from ..inference.utils.get_pipeline_path import get_pipeline_path
from ..utils import logging


def interactive_get_pipeline(pipeline, save_path):
    file_path = get_pipeline_path(pipeline)
    file_name = Path(file_path).name

    if save_path is None:
        logging.info(
            "Please enter the path that you want to save the pipeline config file: (default `./`)"
        )
        target_path = input() or "."
    else:
        target_path = save_path
    target_path = Path(target_path)

    if target_path.suffix not in (".yaml", ".yml"):
        target_path /= file_name

    if not target_path.parent.exists():
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logging.error(f"Failed to create directory: {e}")
            return

    if target_path.exists():
        logging.info(f"The file({target_path}) already exists. Is it covered? (y/N):")
        overwrite = input().lower()
        if overwrite != "y":
            logging.warning("Exit!")
            return
    try:
        shutil.copy2(file_path, target_path)
        logging.info(f"The pipeline config has been saved to: {target_path}")
    except Exception as e:
        logging.error(f"File saving failed: {e}")
