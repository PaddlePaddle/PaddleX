# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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


class TemplateTransforms:
    def __init__(self, mode):
        assert mode in [
            'train', 'eval', 'test'
        ], "Parameter mode in TemplateTransforms should be one of ['train', 'eval', 'test']"
        self.mode = mode

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        assert mode == 'train', "There should be exists augmenters while on train mode"
        self.transforms = augmenters + self.transforms.transforms
