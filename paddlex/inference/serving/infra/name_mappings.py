# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

PIPELINE_APP_ROUTER = {
    "PaddleOCR-VL-1.5": "PaddleOCR-VL",
}


def pipeline_name_to_mod_name(pipeline_name: str) -> str:
    if not pipeline_name:
        raise ValueError("Empty pipeline name")
    if pipeline_name in PIPELINE_APP_ROUTER:
        pipeline_name = PIPELINE_APP_ROUTER[pipeline_name]
    mod_name = pipeline_name.lower().replace("-", "_").replace(".", "")
    if mod_name[0].isdigit():
        return "m_" + mod_name
    return mod_name
