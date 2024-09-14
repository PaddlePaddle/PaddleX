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

from abc import ABC, abstractmethod

from ...utils.subclass_register import AutoRegisterABCMetaClass


def create_pipeline(
    pipeline_name: str,
    model_list: list,
    model_dir_list: list,
    output: str,
    device: str,
) -> "BasePipeline":
    """build model evaluater

    Args:
        pipeline_name (str): the pipeline name, that is name of pipeline class

    Returns:
        BasePipeline: the pipeline, which is subclass of BasePipeline.
    """
    pipeline = BasePipeline.get(pipeline_name)(output=output, device=device)
    pipeline.update_model(model_list, model_dir_list)
    pipeline.load_model()
    return pipeline


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Pipeline"""

    __is_base = True

    # alias the __call__() to predict()
    def __call__(self, *args, **kwargs):
        yield from self.predict(*args, **kwargs)
