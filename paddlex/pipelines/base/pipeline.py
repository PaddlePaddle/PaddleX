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

from ...utils.misc import AutoRegisterABCMetaClass


def build_pipeline(
        pipeline_name: str,
        model_list: list,
        model_dir_list: list,
        output: str,
        device: str, ) -> "BasePipeline":
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
    """Base Pipeline
    """
    __is_base = True

    def __init__(self, disable_print=False, disable_save=False):
        super().__init__()
        self.disable_print = disable_print
        self.disable_save = disable_save

    @abstractmethod
    def load_model(self):
        """load model predictor
        """
        raise NotImplementedError

    @abstractmethod
    def update_model(self, model_name_list, model_dir_list):
        """update model

        Args:
            model_name_list (list): list of model name.
            model_dir_list (list): list of model directory.
        """
        raise NotImplementedError

    @abstractmethod
    def get_input_keys(self):
        """get dict keys of input argument input
        """
        raise NotImplementedError
