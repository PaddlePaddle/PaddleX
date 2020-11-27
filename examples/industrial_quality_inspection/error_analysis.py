# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddlex as pdx

model_dir = 'output/faster_rcnn_r50_vd_dcn/best_model/'
save_dir = 'visualize/faster_rcnn_r50_vd_dcn'
if not osp.exists(save_dir):
    os.makedirs(save_dir)

eval_details_file = osp.join(model_dir, 'eval_details.json')
pdx.det.coco_error_analysis(eval_details_file, save_dir=save_dir)
