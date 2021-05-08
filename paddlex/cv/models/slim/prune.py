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

from functools import partial
import paddle
from paddleslim.dygraph import L1NormFilterPruner, FPGMFilterPruner
from paddlex.cv.transforms import arrange_transforms


def _eval_fn(model, eval_dataset, batch_size=8):
    metric = model.evaluate(eval_dataset, batch_size=batch_size)
    return metric[list(metric.keys())[0]]


def analysis(model,
             dataset,
             batch_size=8,
             criterion='l1_norm',
             save_file='./model.sensi.data'):
    arrange_transforms(
        model_type=model.model_type,
        transforms=dataset.transforms,
        mode='eval')
    if model.model_type == 'segmenter' or model.model_type == 'classifier':
        model.net.train()
        inputs = [1] + list(dataset[0][0].shape)
    elif model.model_type == 'detector':
        inputs = [{
            "image": paddle.ones(
                shape=[1, 3] + list(dataset[0]["image"].shape[:2]),
                dtype='float32'),
            "im_shape": paddle.to_tensor(
                dataset[0]["im_shape"], dtype='float32'),
            "scale_factor": paddle.ones(
                shape=[1, 2], dtype='float32')
        }]
    if criterion == 'l1_norm':
        pruner = L1NormFilterPruner(model.net, inputs=inputs)
    elif criterion == 'fpgm':
        pruner = FPGMFilterPruner(model.net, inputs=inputs)
    sensitivities = pruner.sensitive(
        eval_func=partial(_eval_fn, model, dataset, batch_size),
        sen_file=save_file)

    return sensitivities
