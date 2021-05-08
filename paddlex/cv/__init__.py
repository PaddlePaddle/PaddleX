# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import models
from . import nets
from . import transforms
from . import datasets


# Paddleseg does init_weights() inevitably after building a network, and when
# its parameter `pretrained_model` is None, log info like "No pretrained model to load,
# ResNet_vd will be trained from scratch" will be shown. To prevent this problem,
# we redefine its load_pretrained_model()
def load_pretrained_model(model, pretrained_model):
    pass


from .nets import paddleseg
paddleseg.utils.utils.load_pretrained_model = load_pretrained_model
