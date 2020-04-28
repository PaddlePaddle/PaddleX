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

import argparse
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx


def cal_sensitivies_file(model_dir, dataset, save_file):
    # 加载模型
    model = pdx.load_model(model_dir)

    # 定义验证所用的数据集
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset,
        file_list=os.path.join(dataset, 'val_list.txt'),
        label_list=os.path.join(dataset, 'labels.txt'),
        transforms=model.eval_transforms)

    pdx.slim.cal_params_sensitivities(
        model, save_file, eval_dataset, batch_size=8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        default="./output/yolov3_mobilenet/best_model",
        type=str,
        help="The model path.")
    parser.add_argument(
        "--dataset", default="./insect_det", type=str, help="The model path.")
    parser.add_argument(
        "--save_file",
        default="./sensitivities.data",
        type=str,
        help="The sensitivities file path.")

    args = parser.parse_args()
    cal_sensitivies_file(args.model_dir, args.dataset, args.save_file)
