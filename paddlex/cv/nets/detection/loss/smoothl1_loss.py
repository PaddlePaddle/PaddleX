# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from paddle import fluid

class SmoothL1Loss(object):
    '''
    SmoothL1 Loss
    Args:
        sigma (float): Hyper parameter of smooth L1 loss layer. 

    '''
    def __init__(self, sigma):
        self.sigma = sigma
        
    def __call__(self,
                 x,
                 y,
                 inside_weight=None,
                 outside_weight=None):
        loss_bbox = fluid.layers.smooth_l1(
                x=x,
                y=y,
                inside_weight=inside_weight,
                outside_weight=outside_weight,
                sigma=self.sigma)
        return loss_bbox