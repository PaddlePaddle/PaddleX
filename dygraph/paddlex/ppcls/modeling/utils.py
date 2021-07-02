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

import six
import types
from difflib import SequenceMatcher

from . import architectures


def get_architectures():
    """
    get all of model architectures
    """
    names = []
    for k, v in architectures.__dict__.items():
        if isinstance(v, (types.FunctionType, six.class_types)):
            names.append(k)
    return names


def get_blacklist_model_in_static_mode():
    from paddlex.ppcls.modeling.architectures import distilled_vision_transformer
    from paddlex.ppcls.modeling.architectures import vision_transformer
    blacklist = distilled_vision_transformer.__all__ + vision_transformer.__all__
    return blacklist


def similar_architectures(name='', names=[], thresh=0.1, topk=10):
    """
    inferred similar architectures
    """
    scores = []
    for idx, n in enumerate(names):
        if n.startswith('__'):
            continue
        score = SequenceMatcher(None, n.lower(), name.lower()).quick_ratio()
        if score > thresh:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    similar_names = [names[s[0]] for s in scores[:min(topk, len(scores))]]
    return similar_names
