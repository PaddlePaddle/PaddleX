# copytrue (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import os
import os.path as osp


def prune(best_model_path, dataset_path, sensitivities_path, batch_size):
    import paddlex as pdx
    model = pdx.load_model(best_model_path)

    # build coco dataset
    if osp.exists(osp.join(dataset_path, 'JPEGImages')) and \
        osp.exists(osp.join(dataset_path, 'train.json')) and \
        osp.exists(osp.join(dataset_path, 'val.json')):
        data_dir = osp.join(dataset_path, 'JPEGImages')
        eval_ann_file = osp.join(dataset_path, 'val.json')
        eval_dataset = pdx.datasets.CocoDetection(
            data_dir=data_dir,
            ann_file=eval_ann_file,
            transforms=model.eval_transforms)
    # build voc
    elif osp.exists(osp.join(dataset_path, 'train_list.txt')) and \
        osp.exists(osp.join(dataset_path, 'val_list.txt')) and \
        osp.exists(osp.join(dataset_path, 'labels.txt')):
        eval_file_list = osp.join(dataset_path, 'val_list.txt')
        label_list = osp.join(dataset_path, 'labels.txt')
        eval_dataset = pdx.datasets.VOCDetection(
            data_dir=dataset_path,
            file_list=eval_file_list,
            label_list=label_list,
            transforms=model.eval_transforms)

    pdx.slim.cal_params_sensitivities(model, sensitivities_path, eval_dataset,
                                      batch_size)
