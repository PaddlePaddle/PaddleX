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

import numpy as np

def interpretation_predict(model, images):
    model.arrange_transforms(
            transforms=model.test_transforms, mode='test')
    new_imgs = []
    for i in range(images.shape[0]):
        img = images[i]
        new_imgs.append(model.test_transforms(img)[0])
    new_imgs = np.array(new_imgs)
    result = model.exe.run(
        model.test_prog,
        feed={'image': new_imgs},
        fetch_list=list(model.interpretation_feats.values()))
    return result