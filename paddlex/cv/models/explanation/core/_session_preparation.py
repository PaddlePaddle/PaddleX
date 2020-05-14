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
import paddle.fluid as fluid
import numpy as np


def paddle_get_fc_weights(var_name="fc_0.w_0"):
    fc_weights = fluid.global_scope().find_var(var_name).get_tensor()
    return np.array(fc_weights)


def paddle_resize(extracted_features, outsize):
    resized_features = fluid.layers.resize_bilinear(extracted_features, outsize)
    return resized_features