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
from collections import OrderedDict
import paddle
import paddle.nn.functional as F
import paddlex
from paddlex.cv.nets.paddleseg import models
from paddlex.cv.nets.paddleseg.models import losses
from paddlex.cv.transforms import arrange_transforms
from paddlex.utils import get_single_card_bs
import paddlex.utils.logging as logging
from .base import BaseModel
from .utils import seg_metrics as metrics


class UNet(BaseModel):
    def __init__(self, num_classes=2, use_deconv=False):
        self.init_params = locals()
        super(UNet, self).__init__('segmenter')
        self.num_classes = num_classes
        self.use_deconv = use_deconv
        self.loss_type = 'CrossEntropyLoss'
        self.labels = None

    def build_net(self):
        net = models.__dict__[self.__class__.__name__](
            num_classes=self.num_classes, use_deconv=self.use_deconv)
        test_inputs = [
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ]
        return net, test_inputs

    @staticmethod
    def _preprocess(image, transforms, model_type):
        arrange_transforms(
            model_type=model_type, transforms=transforms, mode='test')
        im = transforms(image)[0]
        ori_shape = im.shape[1:3]
        im = im[np.newaxis, ...]
        return im, ori_shape

    @staticmethod
    def get_reverse_list(ori_shape, transforms):
        reverse_list = []
        h, w = ori_shape[0], ori_shape[1]
        for op in transforms:
            if op.__class__.__name__ in ['Resize']:
                reverse_list.append(('resize', (h, w)))
                h, w = op.height, op.width
        return reverse_list

    @staticmethod
    def _postprocess(pred, origin_shape, transforms):
        reverse_list = UNet.get_reverse_list(origin_shape, transforms)
        for item in reverse_list[::-1]:
            # TODO: 替换成cv2的interpolate（部署阶段无法使用paddle op）
            if item[0] == 'resize':
                h, w = item[1][0], item[1][1]
                pred = F.interpolate(pred, (h, w), mode='nearest')
        return pred

    def run(self, net, inputs, mode):
        net_out = net(inputs[0])
        outputs = OrderedDict()
        if mode == 'test':
            im = paddle.to_tensor(inputs[0])
            net_out = net(im)
            logit = net_out[0]
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            origin_shape = inputs[1]
            pred = UNet._postprocess(pred, origin_shape, transforms=inputs[2])
            pred = paddle.squeeze(pred)
            outputs = {'pred': pred}
        if mode == 'eval':
            net_out = net(inputs[0])
            logit = net_out[0]
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            label = inputs[1]
            origin_shape = label.shape[-2:]
            # TODO: 替换cv2后postprocess移出run
            pred = UNet._postprocess(pred, origin_shape, transforms=inputs[2])

            intersect_area, pred_area, label_area = metrics.calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
        if mode == 'train':
            compute_loss = losses.CrossEntropyLoss()
            loss = compute_loss(net_out[0], inputs[1])
            outputs['loss'] = loss
        return outputs

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
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrained_weights='CITYSCAPES',
              learning_rate=0.01,
              lr_decay_power=0.9):
        self.labels = train_dataset.labels

        self.net, self.test_inputs = self.build_net()
        if self.optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            self.optimizer = self.default_optimizer(
                self.net.parameters(), learning_rate, num_epochs,
                num_steps_each_epoch, lr_decay_power)
        if pretrained_weights is not None and not osp.exists(
                pretrained_weights):
            if pretrained_weights not in ['CITYSCAPES']:
                logging.warning(
                    "Path of pretrained_weights('{}') does not exist!".format(
                        pretrained_weights))
                logging.warning(
                    "Pretrained_weights will be forced to set as 'CITYSCAPES'. "
                    + "If don't want to use pretrained weights, " +
                    "set pretrained_weights to be None.")
                pretrained_weights = 'CITYSCAPES'
        pretrained_dir = osp.join(save_dir, 'pretrain')
        pretrained_dir = osp.join(pretrained_dir, self.__class__.__name__)
        self.net_initialize(
            pretrained_weights=pretrained_weights, save_dir=pretrained_dir)

        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir)

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
                "is forced to be set to {}.".format(batch_size))
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        if return_details:
            eval_details = list()
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
        self.net.eval()
        im, origin_shape = UNet._preprocess(img_file, transforms,
                                            self.model_type)
        data = (im, origin_shape, transforms.transforms)
        outputs = self.run(self.net, data, 'test')
        pred = outputs['pred']
        pred = pred.numpy().astype('uint8')
        return {'label_map', pred}
