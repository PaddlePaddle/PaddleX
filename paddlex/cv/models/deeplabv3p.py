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
import os.path as osp
import numpy as np
import tqdm
import math
import cv2
from multiprocessing.pool import ThreadPool
import paddle.fluid as fluid
import paddlex.utils.logging as logging
import paddlex
from paddlex.cv.transforms import arrange_transforms
from paddlex.cv.datasets import generate_minibatch
from paddlex.cv.transforms.seg_transforms import Compose
from collections import OrderedDict
from .base import BaseAPI
from .utils.seg_eval import ConfusionMatrix
from .utils.visualize import visualize_segmentation


class DeepLabv3p(BaseAPI):
    """实现DeepLabv3+网络的构建并进行训练、评估、预测和模型导出。

    Args:
        num_classes (int): 类别数。
        backbone (str): DeepLabv3+的backbone网络，实现特征图的计算，取值范围为['Xception65', 'Xception41',
            'MobileNetV2_x0.25', 'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5',
            'MobileNetV2_x2.0', 'MobileNetV3_large_x1_0_ssld']。默认'MobileNetV2_x1.0'。
        output_stride (int): backbone 输出特征图相对于输入的下采样倍数，一般取值为8或16。默认16。
        aspp_with_sep_conv (bool):  在asspp模块是否采用separable convolutions。默认True。
        decoder_use_sep_conv (bool)： decoder模块是否采用separable convolutions。默认True。
        encoder_with_aspp (bool): 是否在encoder阶段采用aspp模块。默认True。
        enable_decoder (bool): 是否使用decoder模块。默认True。
        use_bce_loss (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。默认False。
        use_dice_loss (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用，
            当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。默认False。
        class_weight (list/str): 交叉熵损失函数各类损失的权重。当class_weight为list的时候，长度应为
            num_classes。当class_weight为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重
            自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None时，各类的权重1，
            即平时使用的交叉熵损失函数。
        ignore_index (int): label上忽略的值，label为ignore_index的像素不参与损失函数的计算。默认255。
        pooling_crop_size (list): 当backbone为MobileNetV3_large_x1_0_ssld时，需设置为训练过程中模型输入大小, 格式为[W, H]。
            在encoder模块中获取图像平均值时被用到，若为None，则直接求平均值；若为模型输入大小，则使用'pool'算子得到平均值。
            默认值为None。
        input_channel (int): 输入图像通道数。默认值3。

    Raises:
        ValueError: use_bce_loss或use_dice_loss为真且num_calsses > 2。
        ValueError: backbone取值不在['Xception65', 'Xception41', 'MobileNetV2_x0.25',
            'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5', 'MobileNetV2_x2.0', 'MobileNetV3_large_x1_0_ssld']之内。
        ValueError: class_weight为list, 但长度不等于num_class。
                class_weight为str, 但class_weight.low()不等于dynamic。
        TypeError: class_weight不为None时，其类型不是list或str。
    """

    def __init__(self,
                 num_classes=2,
                 backbone='MobileNetV2_x1.0',
                 output_stride=16,
                 aspp_with_sep_conv=True,
                 decoder_use_sep_conv=True,
                 encoder_with_aspp=True,
                 enable_decoder=True,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 pooling_crop_size=None,
                 input_channel=3):
        self.init_params = locals()
        super(DeepLabv3p, self).__init__('segmenter')
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        self.output_stride = output_stride

        if backbone not in [
                'Xception65', 'Xception41', 'MobileNetV2_x0.25',
                'MobileNetV2_x0.5', 'MobileNetV2_x1.0', 'MobileNetV2_x1.5',
                'MobileNetV2_x2.0', 'MobileNetV3_large_x1_0_ssld'
        ]:
            raise ValueError(
                "backbone: {} is set wrong. it should be one of "
                "('Xception65', 'Xception41', 'MobileNetV2_x0.25', 'MobileNetV2_x0.5',"
                " 'MobileNetV2_x1.0', 'MobileNetV2_x1.5', 'MobileNetV2_x2.0', 'MobileNetV3_large_x1_0_ssld')".
                format(backbone))

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

        self.backbone = backbone
        self.num_classes = num_classes
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.aspp_with_sep_conv = aspp_with_sep_conv
        self.decoder_use_sep_conv = decoder_use_sep_conv
        self.encoder_with_aspp = encoder_with_aspp
        self.enable_decoder = enable_decoder
        self.labels = None
        self.sync_bn = True
        self.fixed_input_shape = None
        self.pooling_stride = [1, 1]
        self.pooling_crop_size = pooling_crop_size
        self.aspp_with_se = False
        self.se_use_qsigmoid = False
        self.aspp_convs_filters = 256
        self.aspp_with_concat_projection = True
        self.add_image_level_feature = True
        self.use_sum_merge = False
        self.conv_filters = 256
        self.output_is_logits = False
        self.backbone_lr_mult_list = None
        if 'MobileNetV3' in backbone:
            self.output_stride = 32
            self.pooling_stride = (4, 5)
            self.aspp_with_se = True
            self.se_use_qsigmoid = True
            self.aspp_convs_filters = 128
            self.aspp_with_concat_projection = False
            self.add_image_level_feature = False
            self.use_sum_merge = True
            self.output_is_logits = True
            if self.output_is_logits:
                self.conv_filters = self.num_classes
            self.backbone_lr_mult_list = [0.15, 0.35, 0.65, 0.85, 1]
        self.input_channel = input_channel

    def _get_backbone(self, backbone):
        def mobilenetv2(backbone):
            # backbone: xception结构配置
            # output_stride：下采样倍数
            # end_points: mobilenetv2的block数
            # decode_point: 从mobilenetv2中引出分支所在block数, 作为decoder输入
            if '0.25' in backbone:
                scale = 0.25
            elif '0.5' in backbone:
                scale = 0.5
            elif '1.0' in backbone:
                scale = 1.0
            elif '1.5' in backbone:
                scale = 1.5
            elif '2.0' in backbone:
                scale = 2.0
            end_points = 18
            decode_points = 4
            return paddlex.cv.nets.MobileNetV2(
                scale=scale,
                output_stride=self.output_stride,
                end_points=end_points,
                decode_points=decode_points)

        def xception(backbone):
            # decode_point: 从Xception中引出分支所在block数，作为decoder输入
            # end_point：Xception的block数
            if '65' in backbone:
                decode_points = 2
                end_points = 21
                layers = 65
            if '41' in backbone:
                decode_points = 2
                end_points = 13
                layers = 41
            if '71' in backbone:
                decode_points = 3
                end_points = 23
                layers = 71
            return paddlex.cv.nets.Xception(
                layers=layers,
                output_stride=self.output_stride,
                end_points=end_points,
                decode_points=decode_points)

        def mobilenetv3(backbone):
            scale = 1.0
            lr_mult_list = self.backbone_lr_mult_list
            return paddlex.cv.nets.MobileNetV3(
                scale=scale,
                model_name='large',
                output_stride=self.output_stride,
                lr_mult_list=lr_mult_list,
                for_seg=True)

        if 'Xception' in backbone:
            return xception(backbone)
        elif 'MobileNetV2' in backbone:
            return mobilenetv2(backbone)
        elif 'MobileNetV3' in backbone:
            return mobilenetv3(backbone)

    def build_net(self, mode='train'):
        model = paddlex.cv.nets.segmentation.DeepLabv3p(
            self.num_classes,
            mode=mode,
            backbone=self._get_backbone(self.backbone),
            output_stride=self.output_stride,
            aspp_with_sep_conv=self.aspp_with_sep_conv,
            decoder_use_sep_conv=self.decoder_use_sep_conv,
            encoder_with_aspp=self.encoder_with_aspp,
            enable_decoder=self.enable_decoder,
            use_bce_loss=self.use_bce_loss,
            use_dice_loss=self.use_dice_loss,
            class_weight=self.class_weight,
            ignore_index=self.ignore_index,
            fixed_input_shape=self.fixed_input_shape,
            pooling_stride=self.pooling_stride,
            pooling_crop_size=self.pooling_crop_size,
            aspp_with_se=self.aspp_with_se,
            se_use_qsigmoid=self.se_use_qsigmoid,
            aspp_convs_filters=self.aspp_convs_filters,
            aspp_with_concat_projection=self.aspp_with_concat_projection,
            add_image_level_feature=self.add_image_level_feature,
            use_sum_merge=self.use_sum_merge,
            conv_filters=self.conv_filters,
            output_is_logits=self.output_is_logits,
            input_channel=self.input_channel)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict()
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs['loss'] = model_out
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
                regularization_coeff=4e-05))
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
            train_batch_size (int): 训练数据batch大小。同时作为验证数据batch大小。默认为2。
            eval_dataset (paddlex.datasets): 评估数据读取器。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为2。
            save_dir (str): 模型保存路径。默认'output'。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，
                则自动下载在ImageNet图片数据上预训练的模型权重；若为字符串'COCO'，
                则自动下载在COCO数据集上预训练的模型权重；若为字符串'CITYSCAPES'，
                则自动下载在CITYSCAPES数据集上预训练的模型权重；若为None，则不使用预训练模型。默认'IMAGENET。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认的优化器：使用
                fluid.optimizer.Momentum优化方法，polynomial的学习率衰减策略。
            learning_rate (float): 默认优化器的初始学习率。默认0.01。
            lr_decay_power (float): 默认优化器学习率衰减指数。默认0.9。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认False。
            sensitivities_file (str): 若指定为路径时，则加载路径下敏感度信息进行裁剪；若为字符串'DEFAULT'，
                则自动下载在Cityscapes图片数据上获得的敏感度信息进行裁剪；若为None，则不进行裁剪。默认为None。
            eval_metric_loss (float): 可容忍的精度损失。默认为0.05。
            early_stop (bool): 是否使用提前终止训练策略。默认值为False。
            early_stop_patience (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内
                连续下降或持平，则终止训练。默认值为5。
            resume_checkpoint (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

        Raises:
            ValueError: 模型从inference model进行加载。
        """
        if not self.trainable:
            raise ValueError("Model is not trainable from load_model method.")

        self.labels = train_dataset.labels

        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            optimizer = self.default_optimizer(
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                num_steps_each_epoch=num_steps_each_epoch,
                lr_decay_power=lr_decay_power)

        self.optimizer = optimizer
        # 构建训练、验证、预测网络
        self.build_program()
        # 初始化网络权重
        self.net_initialize(
            startup_prog=fluid.default_startup_program(),
            pretrain_weights=pretrain_weights,
            save_dir=save_dir,
            sensitivities_file=sensitivities_file,
            eval_metric_loss=eval_metric_loss,
            resume_checkpoint=resume_checkpoint)
        # 训练
        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            use_vdl=use_vdl,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)

    def evaluate(self,
                 eval_dataset,
                 batch_size=1,
                 epoch_id=None,
                 return_details=False):
        """评估。

        Args:
            eval_dataset (paddlex.datasets): 评估数据读取器。
            batch_size (int): 评估时的batch大小。默认1。
            epoch_id (int): 当前评估模型所在的训练轮数。
            return_details (bool): 是否返回详细信息。默认False。

        Returns:
            dict: 当return_details为False时，返回dict。包含关键字：'miou'、'category_iou'、'macc'、
                'category_acc'和'kappa'，分别表示平均iou、各类别iou、平均准确率、各类别准确率和kappa系数。
            tuple (metrics, eval_details)：当return_details为True时，增加返回dict (eval_details)，
                包含关键字：'confusion_matrix'，表示评估的混淆矩阵。
        """
        arrange_transforms(
            model_type=self.model_type,
            class_name=self.__class__.__name__,
            transforms=eval_dataset.transforms,
            mode='eval')
        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        conf_mat = ConfusionMatrix(self.num_classes, streaming=True)
        data_generator = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)
        if not hasattr(self, 'parallel_test_prog'):
            with fluid.scope_guard(self.scope):
                self.parallel_test_prog = fluid.CompiledProgram(
                    self.test_prog).with_data_parallel(
                        share_vars_from=self.parallel_train_prog)
        logging.info(
            "Start to evaluating(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(
                enumerate(data_generator()), total=total_steps):
            images = np.array([d[0] for d in data])
            im_info = [d[1] for d in data]
            labels = [d[2] for d in data]

            num_samples = images.shape[0]
            if num_samples < batch_size:
                num_pad_samples = batch_size - num_samples
                pad_images = np.tile(images[0:1], (num_pad_samples, 1, 1, 1))
                images = np.concatenate([images, pad_images])
            feed_data = {'image': images}
            with fluid.scope_guard(self.scope):
                outputs = self.exe.run(
                    self.parallel_test_prog,
                    feed=feed_data,
                    fetch_list=list(self.test_outputs.values()),
                    return_numpy=True)
            pred = outputs[0]
            if num_samples < batch_size:
                pred = pred[0:num_samples]

            for i in range(num_samples):
                one_pred = np.squeeze(pred[i]).astype('uint8')
                one_label = labels[i]
                for info in im_info[i][::-1]:
                    if info[0] == 'resize':
                        w, h = info[1][1], info[1][0]
                        one_pred = cv2.resize(one_pred, (w, h),
                                              cv2.INTER_NEAREST)
                    elif info[0] == 'padding':
                        w, h = info[1][1], info[1][0]
                        one_pred = one_pred[0:h, 0:w]
                one_pred = one_pred.astype('int64')
                one_pred = one_pred[np.newaxis, :, :, np.newaxis]
                one_label = one_label[np.newaxis, np.newaxis, :, :]
                mask = one_label != self.ignore_index
                conf_mat.calculate(pred=one_pred, label=one_label, ignore=mask)
            _, iou = conf_mat.mean_iou()
            logging.debug("[EVAL] Epoch={}, Step={}/{}, iou={}".format(
                epoch_id, step + 1, total_steps, iou))

        category_iou, miou = conf_mat.mean_iou()
        category_acc, oacc = conf_mat.accuracy()
        category_f1score = conf_mat.f1_score()

        metrics = OrderedDict(
            zip([
                'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
                'category_F1-score'
            ], [
                miou, category_iou, oacc, category_acc, conf_mat.kappa(),
                category_f1score
            ]))
        if return_details:
            eval_details = {
                'confusion_matrix': conf_mat.confusion_matrix.tolist()
            }
            return metrics, eval_details
        return metrics

    @staticmethod
    def _preprocess(images,
                    transforms,
                    model_type,
                    class_name,
                    thread_pool=None,
                    input_channel=3):
        arrange_transforms(
            model_type=model_type,
            class_name=class_name,
            transforms=transforms,
            mode='test',
            input_channel=input_channel)
        if thread_pool is not None:
            batch_data = thread_pool.map(transforms, images)
        else:
            batch_data = list()
            for image in images:
                batch_data.append(transforms(image))
        padding_batch = generate_minibatch(batch_data)
        im = np.array(
            [data[0] for data in padding_batch],
            dtype=padding_batch[0][0].dtype)
        im_info = [data[1] for data in padding_batch]
        return im, im_info

    @staticmethod
    def _postprocess(results, im_info):
        pred_list = list()
        logit_list = list()
        for i, (pred, logit) in enumerate(zip(results[0], results[1])):
            pred = pred.astype('uint8')
            pred = np.squeeze(pred).astype('uint8')
            logit = np.transpose(logit, (1, 2, 0))
            for info in im_info[i][::-1]:
                if info[0] == 'resize':
                    w, h = info[1][1], info[1][0]
                    pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
                    logit = cv2.resize(logit, (w, h), cv2.INTER_LINEAR)
                elif info[0] == 'padding':
                    w, h = info[1][1], info[1][0]
                    pred = pred[0:h, 0:w]
                    logit = logit[0:h, 0:w, :]
            pred_list.append(pred)
            logit_list.append(logit)

        preds = list()
        for pred, logit in zip(pred_list, logit_list):
            preds.append({'label_map': pred, 'score_map': logit})
        return preds

    def predict(self, img_file, transforms=None):
        """预测。
        Args:
            img_file(str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
            transforms(paddlex.cv.transforms): 数据预处理操作。

        Returns:
            dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            raise Exception("img_file must be str/np.ndarray")

        if transforms is None:
            transforms = self.test_transforms
        input_channel = getattr(self, 'input_channel', 3)
        im, im_info = DeepLabv3p._preprocess(
            images,
            transforms,
            self.model_type,
            self.__class__.__name__,
            input_channel=input_channel)

        with fluid.scope_guard(self.scope):
            result = self.exe.run(self.test_prog,
                                  feed={'image': im},
                                  fetch_list=list(self.test_outputs.values()),
                                  use_program_cache=True)

        preds = DeepLabv3p._postprocess(result, im_info)
        return preds[0]

    def batch_predict(self, img_file_list, transforms=None):
        """预测。
        Args:
            img_file_list(list|tuple): 对列表（或元组）中的图像同时进行预测，列表中的元素可以是图像路径
                也可以是解码后的排列格式为（H，W，C）且类型为float32且为BGR格式的数组。
            transforms(paddlex.cv.transforms): 数据预处理操作。

        Returns:
            list: 每个元素都为列表，表示各图像的预测结果。各图像的预测结果用字典表示，包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if not isinstance(img_file_list, (list, tuple)):
            raise Exception("im_file must be list/tuple")
        if transforms is None:
            transforms = self.test_transforms
        input_channel = getattr(self, 'input_channel', 3)
        im, im_info = DeepLabv3p._preprocess(
            img_file_list,
            transforms,
            self.model_type,
            self.__class__.__name__,
            self.thread_pool,
            input_channel=input_channel)

        with fluid.scope_guard(self.scope):
            result = self.exe.run(self.test_prog,
                                  feed={'image': im},
                                  fetch_list=list(self.test_outputs.values()),
                                  use_program_cache=True)

        preds = DeepLabv3p._postprocess(result, im_info)
        return preds

    def overlap_tile_predict(self,
                             img_file,
                             tile_size=[512, 512],
                             pad_size=[64, 64],
                             batch_size=32,
                             transforms=None):
        """有重叠的大图切小图预测。
        Args:
            img_file(str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
            tile_size(list|tuple): 滑动窗口的大小，该区域内用于拼接预测结果，格式为（W，H）。默认值为[512, 512]。
            pad_size(list|tuple): 滑动窗口向四周扩展的大小，扩展区域内不用于拼接预测结果，格式为（W，H）。默认值为[64，64]。
            batch_size(int)：对窗口进行批量预测时的批量大小。默认值为32
            transforms(paddlex.cv.transforms): 数据预处理操作。


        Returns:
            dict: 包含关键字'label_map'和'score_map', 'label_map'存储预测结果灰度图，
                像素值表示对应的类别，'score_map'存储各类别的概率，shape=(h, w, num_classes)
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")

        if isinstance(img_file, str):
            image, _ = Compose.decode_image(img_file, None)
        elif isinstance(img_file, np.ndarray):
            image = img_file.copy()
        else:
            raise Exception("im_file must be list/tuple")

        height, width, channel = image.shape
        image_tile_list = list()

        # Padding along the left and right sides
        if pad_size[0] > 0:
            left_pad = cv2.flip(image[0:height, 0:pad_size[0], :], 1)
            right_pad = cv2.flip(image[0:height, -pad_size[0]:width, :], 1)
            padding_image = cv2.hconcat([left_pad, image])
            padding_image = cv2.hconcat([padding_image, right_pad])
        else:
            import copy
            padding_image = copy.deepcopy(image)

        # Padding along the upper and lower sides
        padding_height, padding_width, _ = padding_image.shape
        if pad_size[1] > 0:
            upper_pad = cv2.flip(
                padding_image[0:pad_size[1], 0:padding_width, :], 0)
            lower_pad = cv2.flip(
                padding_image[-pad_size[1]:padding_height, 0:padding_width, :],
                0)
            padding_image = cv2.vconcat([upper_pad, padding_image])
            padding_image = cv2.vconcat([padding_image, lower_pad])

        # crop the padding image into tile pieces
        padding_height, padding_width, _ = padding_image.shape

        for h_id in range(0, height // tile_size[1] + 1):
            for w_id in range(0, width // tile_size[0] + 1):
                left = w_id * tile_size[0]
                upper = h_id * tile_size[1]
                right = min(left + tile_size[0] + pad_size[0] * 2,
                            padding_width)
                lower = min(upper + tile_size[1] + pad_size[1] * 2,
                            padding_height)
                image_tile = padding_image[upper:lower, left:right, :]
                image_tile_list.append(image_tile)

        # predict
        label_map = np.zeros((height, width), dtype=np.uint8)
        score_map = np.zeros(
            (height, width, self.num_classes), dtype=np.float32)
        num_tiles = len(image_tile_list)
        for i in range(0, num_tiles, batch_size):
            begin = i
            end = min(i + batch_size, num_tiles)
            res = self.batch_predict(
                img_file_list=image_tile_list[begin:end],
                transforms=transforms)
            for j in range(begin, end):
                h_id = j // (width // tile_size[0] + 1)
                w_id = j % (width // tile_size[0] + 1)
                left = w_id * tile_size[0]
                upper = h_id * tile_size[1]
                right = min((w_id + 1) * tile_size[0], width)
                lower = min((h_id + 1) * tile_size[1], height)
                tile_label_map = res[j - begin]["label_map"]
                tile_score_map = res[j - begin]["score_map"]
                tile_upper = pad_size[1]
                tile_lower = tile_label_map.shape[0] - pad_size[1]
                tile_left = pad_size[0]
                tile_right = tile_label_map.shape[1] - pad_size[0]
                label_map[upper:lower, left:right] = \
                    tile_label_map[tile_upper:tile_lower, tile_left:tile_right]
                score_map[upper:lower, left:right, :] = \
                    tile_score_map[tile_upper:tile_lower, tile_left:tile_right, :]
        result = {"label_map": label_map, "score_map": score_map}
        return result
