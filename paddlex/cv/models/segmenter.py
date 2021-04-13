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

import os.path as osp
import numpy as np
import cv2
from collections import OrderedDict
import paddle
import paddle.nn.functional as F
import paddlex
from paddlex.cv.nets.paddleseg import models
from paddlex.cv.transforms import arrange_transforms
from paddlex.utils import get_single_card_bs
import paddlex.utils.logging as logging
from .base import BaseModel
from .utils import seg_metrics as metrics
from .utils import pretrain_weights_dict
from paddlex.cv.nets.paddleseg.cvlibs import manager
from paddlex.cv.transforms import Decode

__all__ = ["UNet", "DeepLabV3P", "FastSCNN", "HRNet", "BiSeNetV2"]


class BaseSegmenter(BaseModel):
    def __init__(self,
                 model_name,
                 num_classes=2,
                 use_mixed_loss=False,
                 **params):
        self.init_params = locals()
        super(BaseSegmenter, self).__init__('segmenter')
        if not hasattr(models, model_name):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.losses = None
        self.labels = None
        self.net, self.test_inputs = self.build_net(**params)

    def build_net(self, **params):
        net = models.__dict__[self.model_name](num_classes=self.num_classes,
                                               **params)
        test_inputs = [
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ]
        return net, test_inputs

    def run(self, net, inputs, mode):
        net_out = net(inputs[0])
        logit = net_out[0]
        outputs = OrderedDict()
        if mode == 'test':
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            origin_shape = inputs[1]
            pred = UNet._postprocess(pred, origin_shape, transforms=inputs[2])
            pred = paddle.squeeze(pred)
            outputs = {'pred': pred}
        if mode == 'eval':
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            label = inputs[1]
            origin_shape = [label.shape[-2:]]
            # TODO: 替换cv2后postprocess移出run
            pred = UNet._postprocess(pred, origin_shape, transforms=inputs[2])
            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
        if mode == 'train':
            loss_list = metrics.loss_computation(
                logits_list=net_out, labels=inputs[1], losses=self.losses)
            loss = sum(loss_list)
            outputs['loss'] = loss
        return outputs

    def default_loss(self):
        if isinstance(self.use_mixed_loss, bool):
            if self.use_mixed_loss:
                losses = [
                    manager.LOSSES['CrossEntropyLoss'](),
                    manager.LOSSES['LovaszSoftmaxLoss']()
                ]
                coef = [.8, .2]
                loss_type = [
                    manager.LOSSES['MixedLoss'](losses=losses, coef=coef)
                ]
            else:
                loss_type = [manager.LOSSES['CrossEntropyLoss']()]
        else:
            losses, coef = list(zip(*self.use_mixed_loss))
            if not set(losses).issubset(
                ['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']):
                raise ValueError(
                    "Only 'CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss' are supported."
                )
            losses = [manager.LOSSES[loss]() for loss in losses]
            loss_type = [
                manager.LOSSES['MixedLoss'](losses=losses, coef=list(coef))
            ]
        if self.model_name == 'FastSCNN':
            loss_type *= 2
            loss_coef = [1.0, 0.4]
        elif self.model_name == 'BiSeNetV2':
            loss_type *= 5
            loss_coef = [1.0] * 5
        else:
            loss_coef = [1.0]
        losses = {'types': loss_type, 'coef': loss_coef}
        return losses

    def default_optimizer(self,
                          parameters,
                          learning_rate,
                          num_epochs,
                          num_steps_each_epoch,
                          lr_decay_power=0.9):
        decay_step = num_epochs * num_steps_each_epoch
        lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
            learning_rate, decay_step, end_lr=0, power=lr_decay_power)
        optimizer = paddle.optimizer.Momentum(
            learning_rate=lr_scheduler,
            parameters=parameters,
            momentum=0.9,
            weight_decay=4e-5)
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              optimizer=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='CITYSCAPES',
              learning_rate=0.01,
              lr_decay_power=0.9,
              early_stop=False,
              early_stop_patience=5):
        self.labels = train_dataset.labels
        if self.losses is None:
            self.losses = self.default_loss()
        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            self.optimizer = self.default_optimizer(
                self.net.parameters(), learning_rate, num_epochs,
                num_steps_each_epoch, lr_decay_power)
        else:
            self.optimizer = optimizer
        if pretrain_weights is not None and not osp.exists(pretrain_weights):
            if pretrain_weights not in pretrain_weights_dict[self.model_name]:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                logging.warning("Pretrain_weights is forcibly set to '{}'. "
                                "If don't want to use pretrain weights, "
                                "set pretrain_weights to be None.".format(
                                    pretrain_weights_dict[self.model_name][0]))
                pretrain_weights = pretrain_weights_dict[self.model_name][0]
        pretrained_dir = osp.join(save_dir, 'pretrain')
        self.net_initialize(
            pretrain_weights=pretrain_weights, save_dir=pretrained_dir)

        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)

    def evaluate(self, eval_dataset, batch_size=1, return_details=False):
        arrange_transforms(
            model_type=self.model_type,
            transforms=eval_dataset.transforms,
            mode='eval')

        self.net.eval()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        batch_size_each_card = get_single_card_bs(batch_size)
        if batch_size_each_card > 1:
            batch_size_each_card = 1
            batch_size = batch_size_each_card * paddlex.env_info['num']
            logging.warning(
                "Segmenter supports batch_size=1 for each gpu/cpu card " \
                "only during evaluating, so batch_size " \
                "is forcibly set to {}.".format(batch_size))
        self.eval_data_loader, _ = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader):
                data.append(eval_dataset.transforms.transforms)
                outputs = self.run(self.net, data, 'eval')
                pred_area = outputs['pred_area']
                label_area = outputs['label_area']
                intersect_area = outputs['intersect_area']

                # Gather from all ranks
                if nranks > 1:
                    intersect_area_list = []
                    pred_area_list = []
                    label_area_list = []
                    paddle.distributed.all_gather(intersect_area_list,
                                                  intersect_area)
                    paddle.distributed.all_gather(pred_area_list, pred_area)
                    paddle.distributed.all_gather(label_area_list, label_area)

                    # Some image has been evaluated and should be eliminated in last iter
                    if (step + 1) * nranks > len(eval_dataset):
                        valid = len(eval_dataset) - step * nranks
                        intersect_area_list = intersect_area_list[:valid]
                        pred_area_list = pred_area_list[:valid]
                        label_area_list = label_area_list[:valid]

                    for i in range(len(intersect_area_list)):
                        intersect_area_all = intersect_area_all + intersect_area_list[
                            i]
                        pred_area_all = pred_area_all + pred_area_list[i]
                        label_area_all = label_area_all + label_area_list[i]

                else:
                    intersect_area_all = intersect_area_all + intersect_area
                    pred_area_all = pred_area_all + pred_area
                    label_area_all = label_area_all + label_area
        class_iou, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                                           label_area_all)
        # TODO 确认是按oacc还是macc
        class_acc, oacc = metrics.accuracy(intersect_area_all, pred_area_all)
        kappa = metrics.kappa(intersect_area_all, pred_area_all,
                              label_area_all)
        category_f1score = metrics.f1_score(intersect_area_all, pred_area_all,
                                            label_area_all)
        eval_metrics = OrderedDict(
            zip([
                'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
                'category_F1-score'
            ], [miou, class_iou, oacc, class_acc, kappa, category_f1score]))

        return eval_metrics

    def predict(self, img_file, transforms=None):
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file
        batch_im, batch_origin_shape = BaseSegmenter._preprocess(
            images, transforms, self.model_type)
        self.net.eval()
        data = (batch_im, batch_origin_shape, transforms.transforms)
        outputs = self.run(self.net, data, 'test')
        pred = outputs['pred']
        pred = pred.numpy().astype('uint8')
        return {'label_map': pred}

    @staticmethod
    def _preprocess(images, transforms, model_type):
        arrange_transforms(
            model_type=model_type, transforms=transforms, mode='test')
        batch_im = list()
        batch_ori_shape = list()
        for im in images:
            sample = {'im': im}
            if isinstance(sample['im'], str):
                sample = Decode()(sample)
            ori_shape = sample['im'].shape[:2]
            im = transforms(sample)[0]
            batch_im.append(im)
            batch_ori_shape.append(ori_shape)
        batch_im = paddle.to_tensor(batch_im)

        return batch_im, batch_ori_shape

    @staticmethod
    def get_transforms_shape_info(batch_ori_shape, transforms):
        batch_restore_list = list()
        for ori_shape in batch_ori_shape:
            restore_list = list()
            h, w = ori_shape[0], ori_shape[1]
            for op in transforms:
                if op.__class__.__name__ in ['Resize', 'ResizeByShort']:
                    restore_list.append(('resize', (h, w)))
                    h, w = op.target_h, op.target_w
                if op.__class__.__name__ in ['Padding']:
                    restore_list.append(('padding', (h, w)))
                    h, w = op.target_h, op.target_w
            batch_restore_list.append(restore_list)
        return batch_restore_list

    @staticmethod
    def _postprocess(batch_pred, batch_origin_shape, transforms):
        batch_restore_list = BaseSegmenter.get_transforms_shape_info(
            batch_origin_shape, transforms)
        results = list()
        for pred, restore_list in zip(batch_pred, batch_restore_list):
            pred = paddle.unsqueeze(pred, axis=0)
            for item in restore_list[::-1]:
                # TODO: 替换成cv2的interpolate（部署阶段无法使用paddle op）
                h, w = item[1][0], item[1][1]
                if item[0] == 'resize':
                    pred = F.interpolate(pred, (h, w), mode='nearest')
                elif item[0] == 'padding':
                    pred = pred[:, :, 0:h, 0:w]
                else:
                    pass
            results.append(pred)
        batch_pred = paddle.concat(results, axis=0)
        return batch_pred


class UNet(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 use_deconv=False,
                 align_corners=False):
        params = {'use_deconv': use_deconv, 'align_corners': align_corners}
        super(UNet, self).__init__(
            model_name='UNet',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class DeepLabV3P(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 backbone='ResNet50_vd',
                 use_mixed_loss=False,
                 output_stride=8,
                 backbone_indices=(0, 3),
                 aspp_ratios=(1, 12, 24, 36),
                 aspp_out_channels=256,
                 align_corners=False):
        self.backbone_name = backbone
        if backbone not in ['ResNet50_vd', 'ResNet101_vd']:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50_vd', 'ResNet101_vd')".format(backbone))
        backbone = manager.BACKBONES[backbone](output_stride=output_stride)
        params = {
            'backbone': backbone,
            'backbone_indices': backbone_indices,
            'aspp_ratios': aspp_ratios,
            'aspp_out_channels': aspp_out_channels,
            'align_corners': align_corners
        }
        super(DeepLabV3P, self).__init__(
            model_name='DeepLabV3P',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class FastSCNN(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 align_corners=False):
        params = {'align_corners': align_corners}
        super(FastSCNN, self).__init__(
            model_name='FastSCNN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)


class HRNet(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 width=48,
                 use_mixed_loss=False,
                 align_corners=False):
        if width not in (18, 48):
            raise ValueError(
                "width={} is not supported, please choose from [18, 48]".
                format(width))
        self.backbone_name = 'HRNet_W{}'.format(width)
        backbone = manager.BACKBONES[self.backbone_name](
            align_corners=align_corners)

        params = {'backbone': backbone, 'align_corners': align_corners}
        super(HRNet, self).__init__(
            model_name='FCN',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)
        self.model_name = 'HRNet'


class BiSeNetV2(BaseSegmenter):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 align_corners=False):
        params = {'align_corners': align_corners}
        super(BiSeNetV2, self).__init__(
            model_name='BiSeNetV2',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            **params)
