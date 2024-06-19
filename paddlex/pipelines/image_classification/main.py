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



from ...modules.base import create_model
from ...modules.image_classification.predictor import transforms as T
from ...modules.base.predictor.utils.paddle_inference_predictor import PaddleInferenceOption


class ClsPipeline(object):
    """Cls Pipeline
    """

    def __init__(self,
                 model_name,
                 model_dir=None,
                 output_dir="./output",
                 kernel_option=None):
        self.output_dir = output_dir
        post_transforms = self.get_post_transforms(model_dir)
        kernel_option = self.get_kernel_option(
        ) if kernel_option is None else kernel_option

        self.model = create_model(
            model_name,
            model_dir=model_dir,
            kernel_option=kernel_option,
            post_transforms=post_transforms)

    def __call__(self, input_path):
        return self.model.predict({"input_path": input_path})

    def get_post_transforms(self, model_dir):
        """get post transform ops
        """
        return [T.Topk(topk=1), T.PrintResult()]

    def get_kernel_option(self):
        """get kernel option
        """
        kernel_option = PaddleInferenceOption()
        kernel_option.set_device("gpu")
