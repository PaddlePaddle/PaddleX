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
import numpy as np
import time
import math
import tqdm
import paddle.fluid as fluid
import paddlex.utils.logging as logging
from paddlex.utils import seconds_to_hms
import paddlex
from collections import OrderedDict
from .base import BaseAPI


class BaseClassifier(BaseAPI):
    """构建分类器，并实现其训练、评估、预测和模型导出。
    Args:
        model_name (str): 分类器的模型名字，取值范围为['ResNet18',
                          'ResNet34', 'ResNet50', 'ResNet101',
                          'ResNet50_vd', 'ResNet101_vd', 'DarkNet53',
                          'MobileNetV1', 'MobileNetV2', 'Xception41',
                          'Xception65', 'Xception71']。默认为'ResNet50'。
        num_classes (int): 类别数。默认为1000。
    """

    def __init__(self, model_name='ResNet50', num_classes=1000):
        self.init_params = locals()
        super(BaseClassifier, self).__init__('classifier')
        if not hasattr(paddlex.cv.nets, str.lower(model_name)):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.labels = None
        self.num_classes = num_classes
        self.fixed_input_shape = None

    def build_net(self, mode='train'):
        if self.fixed_input_shape is not None:
            input_shape = [
                None, 3, self.fixed_input_shape[1], self.fixed_input_shape[0]
            ]
            image = fluid.data(
                dtype='float32', shape=input_shape, name='image')
        else:
            image = fluid.data(
                dtype='float32', shape=[None, 3, None, None], name='image')
        if mode != 'test':
            label = fluid.data(dtype='int64', shape=[None, 1], name='label')
        model = getattr(paddlex.cv.nets, str.lower(self.model_name))
        net_out = model(image, num_classes=self.num_classes)
        softmax_out = fluid.layers.softmax(net_out, use_cudnn=False)
        inputs = OrderedDict([('image', image)])
        outputs = OrderedDict([('predict', softmax_out)])
        if mode == 'test':
            self.interpretation_feats = OrderedDict([('logits', net_out)])
        if mode != 'test':
            cost = fluid.layers.cross_entropy(input=softmax_out, label=label)
            avg_cost = fluid.layers.mean(cost)
            acc1 = fluid.layers.accuracy(input=softmax_out, label=label, k=1)
            k = min(5, self.num_classes)
            acck = fluid.layers.accuracy(input=softmax_out, label=label, k=k)
            if mode == 'train':
                self.optimizer.minimize(avg_cost)
            inputs = OrderedDict([('image', image), ('label', label)])
            outputs = OrderedDict([('loss', avg_cost), ('acc1', acc1),
                                   ('acc{}'.format(k), acck)])
        if mode == 'eval':
            del outputs['loss']
        return inputs, outputs

    def default_optimizer(self, learning_rate, lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [
            learning_rate * (lr_decay_gamma**i)
            for i in range(len(lr_decay_epochs) + 1)
        ]
        lr_decay = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
        optimizer = fluid.optimizer.Momentum(
            lr_decay,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-04))
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            train_batch_size (int): 训练数据batch大小。同时作为验证数据batch大小。默认值为64。
            eval_dataset (paddlex.datasets: 验证数据读取器。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代步数）。默认为2。
            save_dir (str): 模型保存路径。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，
                则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：
                fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
            learning_rate (float): 默认优化器的初始学习率。默认为0.025。
            lr_decay_epochs (list): 默认优化器的学习率衰减轮数。默认为[30, 60, 90]。
            lr_decay_gamma (float): 默认优化器的学习率衰减率。默认为0.1。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认值为False。
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
        if not self.trainable:
            raise ValueError("Model is not trainable from load_model method.")
        self.labels = train_dataset.labels
        if optimizer is None:
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            optimizer = self.default_optimizer(
                learning_rate=learning_rate,
                lr_decay_epochs=lr_decay_epochs,
                lr_decay_gamma=lr_decay_gamma,
                num_steps_each_epoch=num_steps_each_epoch)
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
            eval_dataset (paddlex.datasets): 验证数据读取器。
            batch_size (int): 验证数据批大小。默认为1。
            epoch_id (int): 当前评估模型所在的训练轮数。
            return_details (bool): 是否返回详细信息。
        Returns:
          dict: 当return_details为False时，返回dict, 包含关键字：'acc1'、'acc5'，
              分别表示最大值的accuracy、前5个最大值的accuracy。
          tuple (metrics, eval_details): 当return_details为True时，增加返回dict，
              包含关键字：'true_labels'、'pred_scores'，分别代表真实类别id、每个类别的预测得分。
        """
        self.arrange_transforms(
            transforms=eval_dataset.transforms, mode='eval')
        data_generator = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)
        k = min(5, self.num_classes)
        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        true_labels = list()
        pred_scores = list()
        if not hasattr(self, 'parallel_test_prog'):
            self.parallel_test_prog = fluid.CompiledProgram(
                self.test_prog).with_data_parallel(
                    share_vars_from=self.parallel_train_prog)
        batch_size_each_gpu = self._get_single_card_bs(batch_size)
        logging.info(
            "Start to evaluating(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(
                enumerate(data_generator()), total=total_steps):
            images = np.array([d[0] for d in data]).astype('float32')
            labels = [d[1] for d in data]
            num_samples = images.shape[0]
            if num_samples < batch_size:
                num_pad_samples = batch_size - num_samples
                pad_images = np.tile(images[0:1], (num_pad_samples, 1, 1, 1))
                images = np.concatenate([images, pad_images])
            outputs = self.exe.run(self.parallel_test_prog,
                                   feed={'image': images},
                                   fetch_list=list(self.test_outputs.values()))
            outputs = [outputs[0][:num_samples]]
            true_labels.extend(labels)
            pred_scores.extend(outputs[0].tolist())
            logging.debug("[EVAL] Epoch={}, Step={}/{}".format(epoch_id, step +
                                                               1, total_steps))

        pred_top1_label = np.argsort(pred_scores)[:, -1]
        pred_topk_label = np.argsort(pred_scores)[:, -k:]
        acc1 = sum(pred_top1_label == true_labels) / len(true_labels)
        acck = sum(
            [np.isin(x, y)
             for x, y in zip(true_labels, pred_topk_label)]) / len(true_labels)
        metrics = OrderedDict([('acc1', acc1), ('acc{}'.format(k), acck)])
        if return_details:
            eval_details = {
                'true_labels': true_labels,
                'pred_scores': pred_scores
            }
            return metrics, eval_details
        return metrics

    def predict(self, img_file, transforms=None, topk=1):
        """预测。
        Args:
            img_file (str): 预测图像路径。
            transforms (paddlex.cls.transforms): 数据预处理操作。
            topk (int): 预测时前k个最大值。
        Returns:
            list: 其中元素均为字典。字典的关键字为'category_id'、'category'、'score'，
            分别对应预测类别id、预测类别标签、预测得分。
        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        true_topk = min(self.num_classes, topk)
        if transforms is not None:
            self.arrange_transforms(transforms=transforms, mode='test')
            im = transforms(img_file)
        else:
            self.arrange_transforms(
                transforms=self.test_transforms, mode='test')
            im = self.test_transforms(img_file)
        result = self.exe.run(self.test_prog,
                              feed={'image': im},
                              fetch_list=list(self.test_outputs.values()),
                              use_program_cache=True)
        pred_label = np.argsort(result[0][0])[::-1][:true_topk]
        res = [{
            'category_id': l,
            'category': self.labels[l],
            'score': result[0][0][l]
        } for l in pred_label]
        return res


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


class ResNet50_vd(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet50_vd, self).__init__(
            model_name='ResNet50_vd', num_classes=num_classes)


class ResNet101_vd(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet101_vd, self).__init__(
            model_name='ResNet101_vd', num_classes=num_classes)


class ResNet50_vd_ssld(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet50_vd_ssld, self).__init__(
            model_name='ResNet50_vd_ssld', num_classes=num_classes)


class ResNet101_vd_ssld(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ResNet101_vd_ssld, self).__init__(
            model_name='ResNet101_vd_ssld', num_classes=num_classes)


class DarkNet53(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(DarkNet53, self).__init__(
            model_name='DarkNet53', num_classes=num_classes)


class MobileNetV1(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__(
            model_name='MobileNetV1', num_classes=num_classes)


class MobileNetV2(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__(
            model_name='MobileNetV2', num_classes=num_classes)


class MobileNetV3_small(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_small, self).__init__(
            model_name='MobileNetV3_small', num_classes=num_classes)


class MobileNetV3_large(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_large, self).__init__(
            model_name='MobileNetV3_large', num_classes=num_classes)


class MobileNetV3_small_ssld(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_small_ssld, self).__init__(
            model_name='MobileNetV3_small_ssld', num_classes=num_classes)


class MobileNetV3_large_ssld(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_large_ssld, self).__init__(
            model_name='MobileNetV3_large_ssld', num_classes=num_classes)


class Xception65(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(Xception65, self).__init__(
            model_name='Xception65', num_classes=num_classes)


class Xception41(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(Xception41, self).__init__(
            model_name='Xception41', num_classes=num_classes)


class DenseNet121(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(DenseNet121, self).__init__(
            model_name='DenseNet121', num_classes=num_classes)


class DenseNet161(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(DenseNet161, self).__init__(
            model_name='DenseNet161', num_classes=num_classes)


class DenseNet201(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(DenseNet201, self).__init__(
            model_name='DenseNet201', num_classes=num_classes)


class ShuffleNetV2(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(ShuffleNetV2, self).__init__(
            model_name='ShuffleNetV2', num_classes=num_classes)


class HRNet_W18(BaseClassifier):
    def __init__(self, num_classes=1000):
        super(HRNet_W18, self).__init__(
            model_name='HRNet_W18', num_classes=num_classes)
