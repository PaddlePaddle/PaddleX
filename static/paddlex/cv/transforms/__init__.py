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

from . import cls_transforms
from . import det_transforms
from . import seg_transforms

from . import visualize
visualize = visualize.visualize


def build_transforms(model_type, transforms_info, to_rgb=True):
    if model_type == "classifier":
        from . import cls_transforms as T
    elif model_type == "detector":
        from . import det_transforms as T
    elif model_type == "segmenter":
        from . import seg_transforms as T
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise Exception(
                "There's no operator named '{}' in transforms of {}".format(
                    op_name, model_type))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    eval_transforms.to_rgb = to_rgb
    return eval_transforms


def build_transforms_v1(model_type, transforms_info, batch_transforms_info):
    """ 老版本模型加载，仅支持PaddleX前端导出的模型
    """
    logging.debug("Use build_transforms_v1 to reconstruct transforms")
    if model_type == "classifier":
        from . import cls_transforms as T
    elif model_type == "detector":
        from . import det_transforms as T
    elif model_type == "segmenter":
        from . import seg_transforms as T
    transforms = list()
    for op_info in transforms_info:
        op_name = op_info[0]
        op_attr = op_info[1]
        if op_name == 'DecodeImage':
            continue
        if op_name == 'Permute':
            continue
        if op_name == 'ResizeByShort':
            op_attr_new = dict()
            if 'short_size' in op_attr:
                op_attr_new['short_size'] = op_attr['short_size']
            else:
                op_attr_new['short_size'] = op_attr['target_size']
            op_attr_new['max_size'] = op_attr.get('max_size', -1)
            op_attr = op_attr_new
        if op_name.startswith('Arrange'):
            continue
        if not hasattr(T, op_name):
            raise Exception(
                "There's no operator named '{}' in transforms of {}".format(
                    op_name, model_type))
        transforms.append(getattr(T, op_name)(**op_attr))
    if model_type == "detector" and len(batch_transforms_info) > 0:
        op_name = batch_transforms_info[0][0]
        op_attr = batch_transforms_info[0][1]
        assert op_name == "PaddingMiniBatch", "Only PaddingMiniBatch transform is supported for batch transform"
        padding = T.Padding(coarsest_stride=op_attr['coarsest_stride'])
        transforms.append(padding)
    eval_transforms = T.Compose(transforms)
    return eval_transforms


def arrange_transforms(model_type,
                       class_name,
                       transforms,
                       mode='train',
                       input_channel=3):
    transforms.input_channel = input_channel
    # 给transforms添加arrange操作
    if model_type == 'classifier':
        arrange_transform = cls_transforms.ArrangeClassifier
    elif model_type == 'segmenter':
        arrange_transform = seg_transforms.ArrangeSegmenter
    elif model_type == 'detector':
        if class_name == "PPYOLO":
            arrange_name = 'ArrangeYOLOv3'
        else:
            arrange_name = 'Arrange{}'.format(class_name)
        arrange_transform = getattr(det_transforms, arrange_name)
    else:
        raise Exception("Unrecognized model type: {}".format(self.model_type))
    if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
        transforms.transforms[-1] = arrange_transform(mode=mode)
    else:
        transforms.transforms.append(arrange_transform(mode=mode))
