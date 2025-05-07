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


from pathlib import Path


def get_pipeline_path(pipeline_name):
    # XXX: using dict class to handle all pipeline configs
    config_subdir = "configs/pipelines"
    pipeline_path = (
        Path(__file__).parent.parent.parent / config_subdir / f"{pipeline_name}.yaml"
    ).resolve()
    if not Path(pipeline_path).exists():
        return None
    return pipeline_path
