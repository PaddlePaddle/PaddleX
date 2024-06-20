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


from ..base import BasePipeline
from ...modules import create_model, PaddleInferenceOption
from ...modules.object_detection import transforms as T


class InstanceSegPipeline(BasePipeline):
    """InstanceSeg Pipeline
    """
    support_models = "instance_segmentation"

    def __init__(self,
                 model_name=None,
                 model_dir=None,
                 output_dir="./output",
                 kernel_option=None,
                 **kwargs):
        self.model_name = model_name
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.post_transforms = self.get_post_transforms(model_dir)
        self.kernel_option = self.get_kernel_option(
        ) if kernel_option is None else kernel_option
        if self.model_name is not None:
            self.load_model()

    def load_model(self):
        """load model predictor
        """
        assert self.model_name is not None
        self.model = create_model(
            self.model_name,
            model_dir=self.model_dir,
            kernel_option=self.kernel_option,
            post_transforms=self.post_transforms)

    def predict(self, input_path):
        """predict
        """
        return self.model.predict({"input_path": input_path})

    def get_post_transforms(self, model_dir):
        """get post transform ops
        """
        return [T.SaveDetResults(self.output_dir), T.PrintResult()]

    def get_kernel_option(self):
        """get kernel option
        """
        kernel_option = PaddleInferenceOption()
        kernel_option.set_device("gpu")
        return kernel_option

    def update_model_name(self, model_name_list):
        """update model name and re

        Args:
            model_list (list): list of model name.
        """
        assert len(model_name_list) == 1
        self.model_name = model_name_list[0]