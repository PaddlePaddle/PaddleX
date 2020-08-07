#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddlex.seg import transforms
import paddlex as pdx


def train(model_dir, sensitivities_file, eval_metric_loss):
    # 下载和解压视盘分割数据集
    optic_dataset = 'https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz'
    pdx.utils.download_and_decompress(optic_dataset, path='./')

    # 定义训练和验证时的transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.ResizeRangeScaling(),
        transforms.RandomPaddingCrop(crop_size=512), transforms.Normalize()
    ])
    eval_transforms = transforms.Compose([
        transforms.ResizeByLong(long_size=512),
        transforms.Padding(target_size=512), transforms.Normalize()
    ])

    # 定义训练和验证所用的数据集
    train_dataset = pdx.datasets.SegDataset(
        data_dir='optic_disc_seg',
        file_list='optic_disc_seg/train_list.txt',
        label_list='optic_disc_seg/labels.txt',
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.SegDataset(
        data_dir='optic_disc_seg',
        file_list='optic_disc_seg/val_list.txt',
        label_list='optic_disc_seg/labels.txt',
        transforms=eval_transforms)

    if model_dir is None:
        # 使用coco数据集上的预训练权重
        pretrain_weights = "COCO"
    else:
        assert os.path.isdir(model_dir), "Path {} is not a directory".format(
            model_dir)
        pretrain_weights = model_dir
    save_dir = "output/unet"
    if sensitivities_file is not None:
        if sensitivities_file != 'DEFAULT':
            assert os.path.exists(
                sensitivities_file), "Path {} not exist".format(
                    sensitivities_file)
        save_dir = "output/unet_prune"

    num_classes = len(train_dataset.labels)
    model = pdx.seg.UNet(num_classes=num_classes)
    model.train(
        num_epochs=20,
        train_dataset=train_dataset,
        train_batch_size=4,
        eval_dataset=eval_dataset,
        learning_rate=0.01,
        pretrain_weights=pretrain_weights,
        save_dir=save_dir,
        use_vdl=True,
        sensitivities_file=sensitivities_file,
        eval_metric_loss=eval_metric_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir", default=None, type=str, help="The model path.")
    parser.add_argument(
        "--sensitivities_file",
        default=None,
        type=str,
        help="The sensitivities file path.")
    parser.add_argument(
        "--eval_metric_loss",
        default=0.05,
        type=float,
        help="The loss threshold.")

    args = parser.parse_args()
    train(args.model_dir, args.sensitivities_file, args.eval_metric_loss)
