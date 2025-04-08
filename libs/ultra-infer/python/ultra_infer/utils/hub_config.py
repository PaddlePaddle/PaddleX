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
import json
import yaml

from . import hub_env as hubenv


class HubConfig:
    """
    UltraInfer model management configuration class.
    """

    def __init__(self):
        self._initialize()
        self.file = os.path.join(hubenv.CONF_HOME, "config.yaml")

        if not os.path.exists(self.file):
            self.flush()
            return

        with open(self.file, "r") as file:
            try:
                cfg = yaml.load(file, Loader=yaml.FullLoader)
                self.data.update(cfg)
            except:
                ...

    def _initialize(self):
        # Set default configuration values.
        self.data = {}
        self.data["server"] = "http://paddlepaddle.org.cn/paddlehub"

    def reset(self):
        """Reset configuration to default."""
        self._initialize()
        self.flush()

    @property
    def server(self):
        """Model server url."""
        return self.data["server"]

    @server.setter
    def server(self, url: str):
        self.data["server"] = url
        self.flush()

    def flush(self):
        """Flush the current configuration into the configuration file."""
        with open(self.file, "w") as file:
            cfg = json.loads(json.dumps(self.data))
            yaml.dump(cfg, file)

    def __str__(self):
        cfg = json.loads(json.dumps(self.data))
        return yaml.dump(cfg)


config = HubConfig()
