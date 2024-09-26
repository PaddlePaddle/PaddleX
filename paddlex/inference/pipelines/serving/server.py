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

import uvicorn
from fastapi import FastAPI


def run_server(app: FastAPI, *, host: str, port: int, debug: bool) -> None:
    if debug:
        kwargs = {
            "reload": True,
        }
    else:
        kwargs = {}
    uvicorn.run(app, host=host, port=port, log_level="info", **kwargs)
