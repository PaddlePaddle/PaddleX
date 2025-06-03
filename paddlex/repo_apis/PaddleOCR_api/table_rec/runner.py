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


from ...base.utils.subprocess import CompletedProcess
from ..text_rec.runner import TextRecRunner


class TableRecRunner(TextRecRunner):
    """Table Recognition Runner"""

    def predict(
        self, config_path: str, cli_args: list, device: str
    ) -> CompletedProcess:
        """run predicting using dynamic mode

        Args:
            config_path (str): the config file path used to predict.
            cli_args (list): the additional parameters.
            device (str): unused.

        Returns:
            CompletedProcess: the result of predicting subprocess execution.
        """
        cmd = [self.python, "tools/infer_table.py", "-c", config_path]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path: str, cli_args: list, device: str) -> CompletedProcess:
        """run predicting using inference model

        Args:
            config_path (str): the path of config file used to predict.
            cli_args (list): the additional parameters.
            device (str): unused.

        Returns:
            CompletedProcess: the result of inferring subprocess execution.
        """
        cmd = [self.python, "ppstructure/table/predict_structure.py", *cli_args]
        return self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)
