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
import numpy as np
import time
import math
import tqdm
from multiprocessing.pool import ThreadPool
import paddle.fluid as fluid
import paddlex.utils.logging as logging
from paddlex.utils import seconds_to_hms
import paddlex
from paddlex.cv.transforms import arrange_transforms
from paddlex.cv.datasets import generate_minibatch
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
        if self.__class__.__name__ == "AlexNet":
            assert self.fixed_input_shape is not None, "In AlexNet, input_shape should be defined, e.g. model = paddlex.cls.AlexNet(num_classes=1000, input_shape=[224, 224])"
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

    def default_optimizer(self, learning_rate, warmup_steps, warmup_start_lr,
                          lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [
            learning_rate * (lr_decay_gamma**i)
            for i in range(len(lr_decay_epochs) + 1)
        ]
        lr_decay = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
        if warmup_steps > 0:
            if warmup_steps > lr_decay_epochs[0] * num_steps_each_epoch:
                logging.error(
                    "In function train(), parameters should satisfy: warmup_steps <= lr_decay_epochs[0]*num_samples_in_train_dataset",
                    exit=False)
                logging.error(
                    "See this doc for more information: https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/appendix/parameters.md#notice",
                    exit=False)
                logging.error(
                    "warmup_steps should less than {} or lr_decay_epochs[0] greater than {}, please modify 'lr_decay_epochs' or 'warmup_steps' in train function".
                    format(lr_decay_epochs[0] * num_steps_each_epoch,
                           warmup_steps // num_steps_each_epoch))

            lr_decay = fluid.layers.linear_lr_warmup(
                learning_rate=lr_decay,
                warmup_steps=warmup_steps,
                start_lr=warmup_start_lr,
                end_lr=learning_rate)
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
              warmup_steps=0,
              warmup_start_lr=0.0,
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
            warmup_steps(int): 学习率从warmup_start_lr上升至设定的learning_rate，所需的步数，默认为0
            warmup_start_lr(float): 学习率在warmup阶段时的起始值，默认为0.0
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
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
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
        arrange_transforms(
            model_type=self.model_type,
            class_name=self.__class__.__name__,
            transforms=eval_dataset.transforms,
            mode='eval')
        data_generator = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)
        k = min(5, self.num_classes)
        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        true_labels = list()
        pred_scores = list()
        if not hasattr(self, 'parallel_test_prog'):
            with fluid.scope_guard(self.scope):
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
            with fluid.scope_guard(self.scope):
                outputs = self.exe.run(
                    self.parallel_test_prog,
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

    @staticmethod
    def _preprocess(images,
                    transforms,
                    model_type,
                    class_name,
                    thread_pool=None):
        arrange_transforms(
            model_type=model_type,
            class_name=class_name,
            transforms=transforms,
            mode='test')
        if thread_pool is not None:
            batch_data = thread_pool.map(transforms, images)
        else:
            batch_data = list()
            for image in images:
                batch_data.append(transforms(image))
        padding_batch = generate_minibatch(batch_data)
        im = np.array([data[0] for data in padding_batch])

        return im

    @staticmethod
    def _postprocess(results, true_topk, labels):
        preds = list()
        for i, pred in enumerate(results[0]):
            pred_label = np.argsort(pred)[::-1][:true_topk]
            preds.append([{
                'category_id': l,
                'category': labels[l],
                'score': results[0][i][l]
            } for l in pred_label])

        return preds

    def predict(self, img_file, transforms=None, topk=1):
        """预测。
        Args:
            img_file (str|np.ndarray): 预测图像路径，或者是解码后的排列格式为（H, W, C）且类型为float32且为BGR格式的数组。
            transforms (paddlex.cls.transforms): 数据预处理操作。
            topk (int): 预测时前k个最大值。
        Returns:
            list: 其中元素均为字典。字典的关键字为'category_id'、'category'、'score'，
            分别对应预测类别id、预测类别标签、预测得分。
        """

        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        true_topk = min(self.num_classes, topk)
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            raise Exception("img_file must be str/np.ndarray")

        if transforms is None:
            transforms = self.test_transforms
        im = BaseClassifier._preprocess(images, transforms, self.model_type,
                                        self.__class__.__name__)

        with fluid.scope_guard(self.scope):
            result = self.exe.run(self.test_prog,
                                  feed={'image': im},
                                  fetch_list=list(self.test_outputs.values()),
                                  use_program_cache=True)

        preds = BaseClassifier._postprocess(result, true_topk, self.labels)

        return preds[0]

    def batch_predict(self, img_file_list, transforms=None, topk=1):
        """预测。
        Args:
            img_file_list(list|tuple): 对列表（或元组）中的图像同时进行预测，列表中的元素可以是图像路径
                也可以是解码后的排列格式为（H，W，C）且类型为float32且为BGR格式的数组。
            transforms (paddlex.cls.transforms): 数据预处理操作。
            topk (int): 预测时前k个最大值。
        Returns:
            list: 每个元素都为列表，表示各图像的预测结果。在各图像的预测列表中，其中元素均为字典。字典的关键字为'category_id'、'category'、'score'，
            分别对应预测类别id、预测类别标签、预测得分。
        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        true_topk = min(self.num_classes, topk)
        if not isinstance(img_file_list, (list, tuple)):
            raise Exception("im_file must be list/tuple")

        if transforms is None:
            transforms = self.test_transforms
        im = BaseClassifier._preprocess(
            img_file_list, transforms, self.model_type,
            self.__class__.__name__, self.thread_pool)

        with fluid.scope_guard(self.scope):
            result = self.exe.run(self.test_prog,
                                  feed={'image': im},
                                  fetch_list=list(self.test_outputs.values()),
                                  use_program_cache=True)

        preds = BaseClassifier._postprocess(result, true_topk, self.labels)

        return preds


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

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='BAIDU10W',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
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
                则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。若为'BAIDU10W'，则自动下载百度自研10万类预训练。默认为'BAIDU10W'。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：
                fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
            learning_rate (float): 默认优化器的初始学习率。默认为0.025。
            warmup_steps(int): 学习率从warmup_start_lr上升至设定的learning_rate，所需的步数，默认为0
            warmup_start_lr(float): 学习率在warmup阶段时的起始值，默认为0.0
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
        return super(ResNet50_vd, self).train(
            num_epochs, train_dataset, train_batch_size, eval_dataset,
            save_interval_epochs, log_interval_steps, save_dir,
            pretrain_weights, optimizer, learning_rate, warmup_steps,
            warmup_start_lr, lr_decay_epochs, lr_decay_gamma, use_vdl,
            sensitivities_file, eval_metric_loss, early_stop,
            early_stop_patience, resume_checkpoint)


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


class AlexNet(BaseClassifier):
    def __init__(self, num_classes=1000, input_shape=None):
        super(AlexNet, self).__init__(
            model_name='AlexNet', num_classes=num_classes)
        self.fixed_input_shape = input_shape
