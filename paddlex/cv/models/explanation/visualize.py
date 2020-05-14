#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import cv2
import copy
import os.path as osp
import numpy as np
from .core.explanation import Explanation


def visualize(img_file, 
              model, 
              explanation_type='lime',
              num_samples=3000, 
              batch_size=50,
              save_dir='./'):
    model.arrange_transforms(
                transforms=model.test_transforms, mode='test')
    tmp_transforms = copy.deepcopy(model.test_transforms)
    tmp_transforms.transforms = tmp_transforms.transforms[:-2]
    img = tmp_transforms(img_file)[0]
    img = np.around(img).astype('uint8')
    img = np.expand_dims(img, axis=0)
    explaier = None
    if explanation_type == 'lime':
        explaier = get_lime_explaier(img, model, num_samples=num_samples, batch_size=batch_size)
    else:
        raise Exception('The {} explanantion method is not supported yet!'.format(explanation_type))
    img_name = osp.splitext(osp.split(img_file)[-1])[0]
    explaier.explain(img, save_dir=save_dir)
    
    
def get_lime_explaier(img, model, num_samples=3000, batch_size=50):
    def predict_func(image):
        image = image.astype('float32')
        model.test_transforms.transforms = model.test_transforms.transforms[-2:]
        out = model.explanation_predict(image)
        return out[0]
    explaier = Explanation('lime', 
                            predict_func,
                            num_samples=num_samples, 
                            batch_size=batch_size)
    return explaier
    