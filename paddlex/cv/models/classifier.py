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

from __future__ import absolute_import
import os.path as osp
from collections import OrderedDict
import numpy as np
import paddle
from paddle import to_tensor
import paddle.nn.functional as F
from paddlex.utils import logging, TrainingStats
from paddlex.cv.models.base import BaseModel
from paddlex.cv.nets.ppcls.modeling import architectures
from paddlex.cv.nets.ppcls.modeling.loss import CELoss
from paddlex.cv.transforms import arrange_transforms


class BaseClassifier(BaseModel):
    """构建分类器，并实现其训练、评估、预测和模型导出。
    Args:
        model_name (str): 分类器的模型名字，取值范围为['ResNet18',
                          'ResNet34', 'ResNet50', 'ResNet101'。
        num_classes (int): 类别数。默认为1000。
    """

    def __init__(self, model_name='ResNet50', num_classes=1000):
        self.init_params = locals()
        super(BaseClassifier, self).__init__('classifier')
        if not hasattr(architectures, model_name):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.labels = None
        self.num_classes = num_classes

    def build_net(self):
        net = architectures.__dict__[self.model_name](
            class_dim=self.num_classes)
        test_inputs = [
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ]
        return net, test_inputs

    def run(self, net, inputs, mode):
        net_out = net(inputs[0])
        softmax_out = F.softmax(net_out)
        if mode == 'test':
            outputs = OrderedDict([('prediction', softmax_out)])
            self.interpretation_feats = OrderedDict([('logits', net_out)])

        elif mode == 'eval':
            labels = to_tensor(inputs[1].numpy().astype('int64').reshape(-1,
                                                                         1))
            acc1 = paddle.metric.accuracy(softmax_out, label=labels)
            k = min(5, self.num_classes)
            acck = paddle.metric.accuracy(softmax_out, label=labels, k=k)
            # multi cards eval
            if paddle.distributed.get_world_size() > 1:
                acc1 = paddle.distributed.all_reduce(
                    acc1, op=paddle.distributed.ReduceOp.
                    SUM) / paddle.distributed.get_world_size()
                acck = paddle.distributed.all_reduce(
                    acck, op=paddle.distributed.ReduceOp.
                    SUM) / paddle.distributed.get_world_size()

            outputs = OrderedDict([('acc1', acc1), ('acc{}'.format(k), acck),
                                   ('prediction', softmax_out)])

        else:
            # mode == 'train'
            labels = to_tensor(inputs[1].numpy().astype('int64').reshape(-1,
                                                                         1))
            loss = CELoss(class_dim=self.num_classes)
            loss = loss(net_out, inputs[1])
            acc1 = paddle.metric.accuracy(softmax_out, label=labels, k=1)
            k = min(5, self.num_classes)
            acck = paddle.metric.accuracy(softmax_out, label=labels, k=k)

            outputs = OrderedDict([('loss', loss), ('acc1', acc1),
                                   ('acc{}'.format(k), acck)])

        return outputs

    def default_optimizer(self, parameters, learning_rate, warmup_steps,
                          warmup_start_lr, lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [
            learning_rate * (lr_decay_gamma**i)
            for i in range(len(lr_decay_epochs) + 1)
        ]
        scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries, values)
        if warmup_steps > 0:
            if warmup_steps > lr_decay_epochs[0] * num_steps_each_epoch:
                logging.error(
                    "In function train(), parameters should satisfy: "
                    "warmup_steps <= lr_decay_epochs[0]*num_samples_in_train_dataset",
                    exit=False)
                logging.error(
                    "See this doc for more information: "
                    "https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/appendix/parameters.md#notice",
                    exit=False)
                logging.error(
                    "warmup_steps should less than {} or lr_decay_epochs[0] greater than {}, "
                    "please modify 'lr_decay_epochs' or 'warmup_steps' in train function".
                    format(lr_decay_epochs[0] * num_steps_each_epoch,
                           warmup_steps // num_steps_each_epoch))

            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=scheduler,
                warmup_steps=warmup_steps,
                start_lr=warmup_start_lr,
                end_lr=learning_rate)
        optimizer = paddle.optimizer.Momentum(
            scheduler,
            momentum=.9,
            weight_decay=paddle.regularizer.L2Decay(coeff=1e-04),
            parameters=parameters)
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=10,
              save_dir='output',
              pretrained_weights='IMAGENET',
              learning_rate=.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=(30, 60, 90),
              lr_decay_gamma=0.1,
              early_stop=False,
              early_stop_patience=5):
        self.labels = train_dataset.labels

        # build net
        self.net, self.test_inputs = self.build_net()

        # build optimizer if not defined
        if self.optimizer is None:
            num_steps_each_epoch = len(train_dataset) // train_batch_size
            self.optimizer = self.default_optimizer(
                parameters=self.net.parameters(),
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                lr_decay_epochs=lr_decay_epochs,
                lr_decay_gamma=lr_decay_gamma,
                num_steps_each_epoch=num_steps_each_epoch)

        # initiate weights
        if pretrained_weights is not None and not osp.exists(
                pretrained_weights):
            if pretrained_weights not in ['IMAGENET']:
                logging.warning(
                    "Path of pretrained_weights('{}') does not exist!".format(
                        pretrained_weights))
                logging.warning(
                    "Pretrained_weights will be forced to set as 'IMAGENET'. "
                    + "If don't want to use pretrained weights, " +
                    "set pretrained_weights to be None.")
                pretrained_weights = 'IMAGENET'
        pretrained_dir = osp.join(save_dir, 'pretrain')
        self.net_initialize(
            pretrained_weights=pretrained_weights, save_dir=pretrained_dir)

        # start train loop
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

    def evaluate(self,
                 eval_dataset,
                 batch_size=1,
                 epoch_id=None,
                 return_details=False):
        # 给transform添加arrange操作
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
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')
        eval_metrics = TrainingStats()
        eval_details = None
        if return_details:
            eval_details = list()

        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader()):
                outputs = self.run(self.net, data, mode='eval')
                if return_details:
                    eval_details.append(outputs['prediction'].numpy())
                outputs.pop('prediction')
                eval_metrics.update(outputs)

        return eval_metrics.get(), eval_details

    @staticmethod
    def _preprocess(images, transforms, model_type):
        arrange_transforms(
            model_type=model_type, transforms=transforms, mode='test')
        batch_data = list()
        for image in images:
            batch_data.append(transforms(image)[0])

        batch_data = to_tensor(batch_data)

        return batch_data,

    @staticmethod
    def _postprocess(results, true_topk, labels):
        preds = list()
        for i, pred in enumerate(results):
            pred_label = np.argsort(pred)[::-1][:true_topk]
            preds.append([{
                'category_id': l,
                'category': labels[l],
                'score': results[i][l]
            } for l in pred_label])

        return preds

    def predict(self, img_file, transforms=None, topk=1):
        true_topk = min(self.num_classes, topk)
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file
        if transforms is None:
            transforms = self.test_transforms
        im = BaseClassifier._preprocess(images, transforms, self.model_type)
        self.net.eval()
        with paddle.no_grad():
            outputs = self.run(self.net, im, mode='test')
        prediction = outputs['prediction'].numpy()
        prediction = BaseClassifier._postprocess(prediction, true_topk,
                                                 self.labels)

        return prediction


class ResNet18(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__(
            model_name='ResNet18', num_classes=num_classes)


class ResNet34(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__(
            model_name='ResNet34', num_classes=num_classes)


class ResNet50(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__(
            model_name='ResNet50', num_classes=num_classes)


class ResNet101(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet101, self).__init__(
            model_name='ResNet101', num_classes=num_classes)


class ResNet152(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet152, self).__init__(
            model_name='ResNet152', num_classes=num_classes)
