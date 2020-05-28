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

from __future__ import absolute_import
import paddle.fluid as fluid
import paddlex
from collections import OrderedDict
from .deeplabv3p import DeepLabv3p


class HRNet(DeepLabv3p):
    """实现HRNet网络的构建并进行训练、评估、预测和模型导出。

    Args:
        num_classes (int): 类别数。
        width (int): 高分辨率分支中特征层的通道数量。默认值为18。可选择取值为[18, 30, 32, 40, 44, 48, 60, 64]。
        use_bce_loss (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。默认False。
        use_dice_loss (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。
            当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。默认False。
        class_weight (list/str): 交叉熵损失函数各类损失的权重。当class_weight为list的时候，长度应为
            num_classes。当class_weight为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重
            自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，
            即平时使用的交叉熵损失函数。
        ignore_index (int): label上忽略的值，label为ignore_index的像素不参与损失函数的计算。默认255。

    Raises:
        ValueError: use_bce_loss或use_dice_loss为真且num_calsses > 2。
        ValueError: class_weight为list, 但长度不等于num_class。
            class_weight为str, 但class_weight.low()不等于dynamic。
        TypeError: class_weight不为None时，其类型不是list或str。
    """

    def __init__(self,
                 num_classes=2,
                 width=18,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255):
        self.init_params = locals()
        super(DeepLabv3p, self).__init__('segmenter')
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        if class_weight is not None:
            if isinstance(class_weight, list):
                if len(class_weight) != num_classes:
                    raise ValueError(
                        "Length of class_weight should be equal to number of classes"
                    )
            elif isinstance(class_weight, str):
                if class_weight.lower() != 'dynamic':
                    raise ValueError(
                        "if class_weight is string, must be dynamic!")
            else:
                raise TypeError(
                    'Expect class_weight is a list or string but receive {}'.
                    format(type(class_weight)))
        self.num_classes = num_classes
        self.width = width
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.labels = None

    def build_net(self, mode='train'):
        model = paddlex.cv.nets.segmentation.HRNet(
            self.num_classes,
            width=self.width,
            mode=mode,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
        elif mode == 'eval':
            outputs['loss'] = model_out[0]
            outputs['pred'] = model_out[1]
            outputs['label'] = model_out[2]
            outputs['mask'] = model_out[3]
        else:
            outputs['pred'] = model_out[0]
            outputs['logit'] = model_out[1]
        return inputs, outputs

    def default_optimizer(self,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          lr_decay_power=0.9):
        decay_step = num_epochs * num_steps_each_epoch
        lr_decay = fluid.layers.polynomial_decay(
            learning_rate,
            decay_step,
            end_learning_rate=0,
            power=lr_decay_power)
        optimizer = fluid.optimizer.Momentum(
            lr_decay,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=5e-04))
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
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
            train_batch_size (int): 训练数据batch大小。同时作为验证数据batch大小。默认2。
            eval_dataset (paddlex.datasets): 评估数据读取器。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
            save_dir (str): 模型保存路径。默认'output'。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，
                则自动下载在IMAGENET图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
            optimizer (paddle.fluid.optimizer): 优化器。当改参数为None时，使用默认的优化器：使用
                fluid.optimizer.Momentum优化方法，polynomial的学习率衰减策略。
            learning_rate (float): 默认优化器的初始学习率。默认0.01。
            lr_decay_power (float): 默认优化器学习率多项式衰减系数。默认0.9。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认False。
            sensitivities_file (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，
                则自动下载在ImageNet图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
            eval_metric_loss (float): 可容忍的精度损失。默认为0.05。
            early_stop (bool): 是否使用提前终止训练策略。默认值为False。
            early_stop_patience (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内
                连续下降或持平，则终止训练。默认值为5。
            resume_checkpoint (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

        Raises:
            ValueError: 模型从inference model进行加载。
        """
        return super(HRNet, self).train(
            num_epochs, train_dataset, train_batch_size, eval_dataset,
            save_interval_epochs, log_interval_steps, save_dir,
            pretrain_weights, optimizer, learning_rate, lr_decay_power,
            use_vdl, sensitivities_file, eval_metric_loss, early_stop,
            early_stop_patience, resume_checkpoint)
