# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlex_hps_server import BaseTritonPythonModel
from paddlex_hps_server.storage import create_storage

# Do we need a lock?
DEFAULT_INDEX_DIR = ".indexes"


class BaseShiTuModel(BaseTritonPythonModel):
    def initialize(self, args):
        super().initialize(args)
        self.context = {}
        if self.app_config.extra and "index_storage" in self.app_config.extra:
            self.context["index_storage"] = create_storage(
                self.app_config.extra["index_storage"]
            )
        else:
            self.context["index_storage"] = create_storage(
                {"type": "file_system", "directory": DEFAULT_INDEX_DIR}
            )
