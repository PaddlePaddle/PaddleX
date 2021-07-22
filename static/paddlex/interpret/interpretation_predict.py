# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import copy
import paddle.fluid as fluid
from paddlex.cv.transforms import arrange_transforms


def interpretation_predict(model, images):
    images = images.astype('float32')
    arrange_transforms(
        model.model_type,
        model.__class__.__name__,
        transforms=model.test_transforms,
        mode='test')
    tmp_transforms = copy.deepcopy(model.test_transforms.transforms)
    model.test_transforms.transforms = model.test_transforms.transforms[-2:]

    new_imgs = []
    for i in range(images.shape[0]):
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
        new_imgs.append(model.test_transforms(images[i])[0])

    new_imgs = np.array(new_imgs)
    with fluid.scope_guard(model.scope):
        out = model.exe.run(
            model.test_prog,
            feed={'image': new_imgs},
            fetch_list=list(model.interpretation_feats.values()))

    model.test_transforms.transforms = tmp_transforms

    return out
