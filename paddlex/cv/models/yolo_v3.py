# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import math
import tqdm
import paddlex
from .ppyolo import PPYOLO


class YOLOv3(PPYOLO):
    """构建YOLOv3，并实现其训练、评估、预测和模型导出。

    Args:
        num_classes (int): 类别数。默认为80。
        backbone (str): YOLOv3的backbone网络，取值范围为['DarkNet53',
            'ResNet34', 'MobileNetV1', 'MobileNetV3_large']。默认为'MobileNetV1'。
        anchors (list|tuple): anchor框的宽度和高度，为None时表示使用默认值
                    [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                    [59, 119], [116, 90], [156, 198], [373, 326]]。
        anchor_masks (list|tuple): 在计算YOLOv3损失时，使用anchor的mask索引，为None时表示使用默认值
                    [[6, 7, 8], [3, 4, 5], [0, 1, 2]]。
        ignore_threshold (float): 在计算YOLOv3损失时，IoU大于`ignore_threshold`的预测框的置信度被忽略。默认为0.7。
        nms_score_threshold (float): 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.01。
        nms_topk (int): 进行NMS时，根据置信度保留的最大检测框数。默认为1000。
        nms_keep_topk (int): 进行NMS后，每个图像要保留的总检测框数。默认为100。
        nms_iou_threshold (float): 进行NMS时，用于剔除检测框IoU的阈值。默认为0.45。
        label_smooth (bool): 是否使用label smooth。默认值为False。
        train_random_shapes (list|tuple): 训练时从列表中随机选择图像大小。默认值为[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]。
        input_channel (int): 输入图像的通道数量。默认为3。
    """

    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV1',
                 anchors=None,
                 anchor_masks=None,
                 ignore_threshold=0.7,
                 nms_score_threshold=0.01,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 label_smooth=False,
                 train_random_shapes=[
                     320, 352, 384, 416, 448, 480, 512, 544, 576, 608
                 ],
                 input_channel=3):
        self.init_params = locals()
        backbones = [
            'DarkNet53', 'ResNet34', 'MobileNetV1', 'MobileNetV3_large'
        ]
        assert backbone in backbones, "backbone should be one of {}".format(
            backbones)
        super(PPYOLO, self).__init__('detector')
        self.backbone = backbone
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.ignore_threshold = ignore_threshold
        self.nms_score_threshold = nms_score_threshold
        self.nms_topk = nms_topk
        self.nms_keep_topk = nms_keep_topk
        self.nms_iou_threshold = nms_iou_threshold
        self.label_smooth = label_smooth
        self.sync_bn = True
        self.train_random_shapes = train_random_shapes
        self.fixed_input_shape = None
        self.use_fine_grained_loss = False
        self.use_coord_conv = False
        self.use_iou_aware = False
        self.use_spp = False
        self.use_drop_block = False
        self.use_iou_loss = False
        self.scale_x_y = 1.
        self.use_matrix_nms = False
        self.use_ema = False
        self.with_dcn_v2 = False
        self.input_channel = input_channel

    def _get_backbone(self, backbone_name):
        if backbone_name == 'DarkNet53':
            backbone = paddlex.cv.nets.DarkNet(norm_type='sync_bn')
        elif backbone_name == 'ResNet34':
            backbone = paddlex.cv.nets.ResNet(
                norm_type='sync_bn',
                layers=34,
                freeze_norm=False,
                norm_decay=0.,
                feature_maps=[3, 4, 5],
                freeze_at=0)
        elif backbone_name == 'MobileNetV1':
            backbone = paddlex.cv.nets.MobileNetV1(norm_type='sync_bn')
        elif backbone_name.startswith('MobileNetV3'):
            model_name = backbone_name.split('_')[1]
            backbone = paddlex.cv.nets.MobileNetV3(
                norm_type='sync_bn', model_name=model_name)
        return backbone

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=8,
              eval_dataset=None,
              save_interval_epochs=20,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=1.0 / 8000,
              warmup_steps=1000,
              warmup_start_lr=0.0,
              lr_decay_epochs=[213, 240],
              lr_decay_gamma=0.1,
              metric=None,
              use_vdl=False,
              sensitivities_file=None,
              eval_metric_loss=0.05,
              early_stop=False,
              early_stop_patience=5,
              resume_checkpoint=None):
        """训练。

        Args:
            num_epochs (int): 训练迭代轮数。
            train_dataset (paddlex.datasets): 训练数据读取器。
            train_batch_size (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与显卡
                数量之商为验证数据batch大小。默认值为8。
            eval_dataset (paddlex.datasets): 验证数据读取器。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为20。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为10。
            save_dir (str): 模型保存路径。默认值为'output'。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，
                则自动下载在ImageNet图片数据上预训练的模型权重；若为字符串'COCO'，
                则自动下载在COCO数据集上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：
                fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
            learning_rate (float): 默认优化器的学习率。默认为1.0/8000。
            warmup_steps (int):  默认优化器进行warmup过程的步数。默认为1000。
            warmup_start_lr (int): 默认优化器warmup的起始学习率。默认为0.0。
            lr_decay_epochs (list): 默认优化器的学习率衰减轮数。默认为[213, 240]。
            lr_decay_gamma (float): 默认优化器的学习率衰减率。默认为0.1。
            metric (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认值为None。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认值为False。
            sensitivities_file (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，
                则自动下载在ImageNet图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
            eval_metric_loss (float): 可容忍的精度损失。默认为0.05。
            early_stop (bool): 是否使用提前终止训练策略。默认值为False。
            early_stop_patience (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内
                连续下降或持平，则终止训练。默认值为5。
            resume_checkpoint (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

        Raises:
            ValueError: 评估类型不在指定列表中。
            ValueError: 模型从inference model进行加载。
        """

        return super(YOLOv3, self).train(
            num_epochs, train_dataset, train_batch_size, eval_dataset,
            save_interval_epochs, log_interval_steps, save_dir,
            pretrain_weights, optimizer, learning_rate, warmup_steps,
            warmup_start_lr, lr_decay_epochs, lr_decay_gamma, metric, use_vdl,
            sensitivities_file, eval_metric_loss, early_stop,
            early_stop_patience, resume_checkpoint, False)
