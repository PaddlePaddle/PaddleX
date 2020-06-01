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
import math
import tqdm
import numpy as np
import paddle.fluid as fluid
import paddlex.utils.logging as logging
import paddlex
import os.path as osp
import copy
from .base import BaseAPI
from collections import OrderedDict
from .utils.detection_eval import eval_results, bbox2out


class FasterRCNN(BaseAPI):
    """构建FasterRCNN，并实现其训练、评估、预测和模型导出。

    Args:
        num_classes (int): 包含了背景类的类别数。默认为81。
        backbone (str): FasterRCNN的backbone网络，取值范围为['ResNet18', 'ResNet50',
            'ResNet50_vd', 'ResNet101', 'ResNet101_vd', 'HRNet_W18']。默认为'ResNet50'。
        with_fpn (bool): 是否使用FPN结构。默认为True。
        aspect_ratios (list): 生成anchor高宽比的可选值。默认为[0.5, 1.0, 2.0]。
        anchor_sizes (list): 生成anchor大小的可选值。默认为[32, 64, 128, 256, 512]。
    """

    def __init__(self,
                 num_classes=81,
                 backbone='ResNet50',
                 with_fpn=True,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[32, 64, 128, 256, 512]):
        self.init_params = locals()
        super(FasterRCNN, self).__init__('detector')
        backbones = [
            'ResNet18', 'ResNet50', 'ResNet50_vd', 'ResNet101', 'ResNet101_vd',
            'HRNet_W18'
        ]
        assert backbone in backbones, "backbone should be one of {}".format(
            backbones)
        self.backbone = backbone
        self.num_classes = num_classes
        self.with_fpn = with_fpn
        self.aspect_ratios = aspect_ratios
        self.anchor_sizes = anchor_sizes
        self.labels = None
        self.fixed_input_shape = None

    def _get_backbone(self, backbone_name):
        norm_type = None
        if backbone_name == 'ResNet18':
            layers = 18
            variant = 'b'
        elif backbone_name == 'ResNet50':
            layers = 50
            variant = 'b'
        elif backbone_name == 'ResNet50_vd':
            layers = 50
            variant = 'd'
            norm_type = 'affine_channel'
        elif backbone_name == 'ResNet101':
            layers = 101
            variant = 'b'
            norm_type = 'affine_channel'
        elif backbone_name == 'ResNet101_vd':
            layers = 101
            variant = 'd'
            norm_type = 'affine_channel'
        elif backbone_name == 'HRNet_W18':
            backbone = paddlex.cv.nets.hrnet.HRNet(
                width=18, freeze_norm=True, norm_decay=0., freeze_at=0)
            if self.with_fpn is False:
                self.with_fpn = True
            return backbone
        if self.with_fpn:
            backbone = paddlex.cv.nets.resnet.ResNet(
                norm_type='bn' if norm_type is None else norm_type,
                layers=layers,
                variant=variant,
                freeze_norm=True,
                norm_decay=0.,
                feature_maps=[2, 3, 4, 5],
                freeze_at=2)
        else:
            backbone = paddlex.cv.nets.resnet.ResNet(
                norm_type='affine_channel' if norm_type is None else norm_type,
                layers=layers,
                variant=variant,
                freeze_norm=True,
                norm_decay=0.,
                feature_maps=4,
                freeze_at=2)
        return backbone

    def build_net(self, mode='train'):
        train_pre_nms_top_n = 2000 if self.with_fpn else 12000
        test_pre_nms_top_n = 1000 if self.with_fpn else 6000
        model = paddlex.cv.nets.detection.FasterRCNN(
            backbone=self._get_backbone(self.backbone),
            mode=mode,
            num_classes=self.num_classes,
            with_fpn=self.with_fpn,
            aspect_ratios=self.aspect_ratios,
            anchor_sizes=self.anchor_sizes,
            train_pre_nms_top_n=train_pre_nms_top_n,
            test_pre_nms_top_n=test_pre_nms_top_n,
            fixed_input_shape=self.fixed_input_shape)
        inputs = model.generate_inputs()
        if mode == 'train':
            model_out = model.build_net(inputs)
            loss = model_out['loss']
            self.optimizer.minimize(loss)
            outputs = OrderedDict(
                [('loss', model_out['loss']),
                 ('loss_cls', model_out['loss_cls']),
                 ('loss_bbox', model_out['loss_bbox']),
                 ('loss_rpn_cls', model_out['loss_rpn_cls']), (
                     'loss_rpn_bbox', model_out['loss_rpn_bbox'])])
        else:
            outputs = model.build_net(inputs)
        return inputs, outputs

    def default_optimizer(self, learning_rate, warmup_steps, warmup_start_lr,
                          lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        if warmup_steps > lr_decay_epochs[0] * num_steps_each_epoch:
            raise Exception("warmup_steps should less than {}".format(
                lr_decay_epochs[0] * num_steps_each_epoch))
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [(lr_decay_gamma**i) * learning_rate
                  for i in range(len(lr_decay_epochs) + 1)]
        lr_decay = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
        lr_warmup = fluid.layers.linear_lr_warmup(
            learning_rate=lr_decay,
            warmup_steps=warmup_steps,
            start_lr=warmup_start_lr,
            end_lr=learning_rate)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr_warmup,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-04))
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
              learning_rate=0.0025,
              warmup_steps=500,
              warmup_start_lr=1.0 / 1200,
              lr_decay_epochs=[8, 11],
              lr_decay_gamma=0.1,
              metric=None,
              use_vdl=False,
              early_stop=False,
              early_stop_patience=5,
              resume_checkpoint=None):
        """训练。

        Args:
            num_epochs (int): 训练迭代轮数。
            train_dataset (paddlex.datasets): 训练数据读取器。
            train_batch_size (int): 训练数据batch大小。目前检测仅支持单卡评估，训练数据batch大小与
                显卡数量之商为验证数据batch大小。默认为2。
            eval_dataset (paddlex.datasets): 验证数据读取器。
            save_interval_epochs (int): 模型保存间隔（单位：迭代轮数）。默认为1。
            log_interval_steps (int): 训练日志输出间隔（单位：迭代次数）。默认为20。
            save_dir (str): 模型保存路径。默认值为'output'。
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为字符串'IMAGENET'，
                则自动下载在ImageNet图片数据上预训练的模型权重；若为None，则不使用预训练模型。默认为'IMAGENET'。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：
                fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
            learning_rate (float): 默认优化器的初始学习率。默认为0.0025。
            warmup_steps (int):  默认优化器进行warmup过程的步数。默认为500。
            warmup_start_lr (int): 默认优化器warmup的起始学习率。默认为1.0/1200。
            lr_decay_epochs (list): 默认优化器的学习率衰减轮数。默认为[8, 11]。
            lr_decay_gamma (float): 默认优化器的学习率衰减率。默认为0.1。
            metric (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认值为None。
            use_vdl (bool): 是否使用VisualDL进行可视化。默认值为False。
            early_stop (bool): 是否使用提前终止训练策略。默认值为False。
            early_stop_patience (int): 当使用提前终止训练策略时，如果验证集精度在`early_stop_patience`个epoch内
                连续下降或持平，则终止训练。默认值为5。
            resume_checkpoint (str): 恢复训练时指定上次训练保存的模型路径。若为None，则不会恢复训练。默认值为None。

        Raises:
            ValueError: 评估类型不在指定列表中。
            ValueError: 模型从inference model进行加载。
        """
        if metric is None:
            if isinstance(train_dataset, paddlex.datasets.CocoDetection):
                metric = 'COCO'
            elif isinstance(train_dataset, paddlex.datasets.VOCDetection) or \
                    isinstance(train_dataset, paddlex.datasets.EasyDataDet):
                metric = 'VOC'
            else:
                raise ValueError(
                    "train_dataset should be datasets.VOCDetection or datasets.COCODetection or datasets.EasyDataDet."
                )
        assert metric in ['COCO', 'VOC'], "Metric only support 'VOC' or 'COCO'"
        self.metric = metric
        if not self.trainable:
            raise ValueError("Model is not trainable from load_model method.")
        self.labels = copy.deepcopy(train_dataset.labels)
        self.labels.insert(0, 'background')
        # 构建训练网络
        if optimizer is None:
            # 构建默认的优化策略
            num_steps_each_epoch = train_dataset.num_samples // train_batch_size
            optimizer = self.default_optimizer(
                learning_rate, warmup_steps, warmup_start_lr, lr_decay_epochs,
                lr_decay_gamma, num_steps_each_epoch)
        self.optimizer = optimizer
        # 构建训练、验证、测试网络
        self.build_program()
        fuse_bn = True
        if self.with_fpn and self.backbone in [
                'ResNet18', 'ResNet50', 'HRNet_W18'
        ]:
            fuse_bn = False
        self.net_initialize(
            startup_prog=fluid.default_startup_program(),
            pretrain_weights=pretrain_weights,
            fuse_bn=fuse_bn,
            save_dir=save_dir,
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
                 metric=None,
                 return_details=False):
        """评估。

        Args:
            eval_dataset (paddlex.datasets): 验证数据读取器。
            batch_size (int): 验证数据批大小。默认为1。当前只支持设置为1。
            epoch_id (int): 当前评估模型所在的训练轮数。
            metric (bool): 训练过程中评估的方式，取值范围为['COCO', 'VOC']。默认为None，
                根据用户传入的Dataset自动选择，如为VOCDetection，则metric为'VOC';
                如为COCODetection，则metric为'COCO'。
            return_details (bool): 是否返回详细信息。默认值为False。

        Returns:
            tuple (metrics, eval_details) /dict (metrics): 当return_details为True时，返回(metrics, eval_details)，
                当return_details为False时，返回metrics。metrics为dict，包含关键字：'bbox_mmap'或者’bbox_map‘，
                分别表示平均准确率平均值在各个阈值下的结果取平均值的结果（mmAP）、平均准确率平均值（mAP）。
                eval_details为dict，包含关键字：'bbox'，对应元素预测结果列表，每个预测结果由图像id、
                预测框类别id、预测框坐标、预测框得分；’gt‘：真实标注框相关信息。
        """
        self.arrange_transforms(
            transforms=eval_dataset.transforms, mode='eval')
        if metric is None:
            if hasattr(self, 'metric') and self.metric is not None:
                metric = self.metric
            else:
                if isinstance(eval_dataset, paddlex.datasets.CocoDetection):
                    metric = 'COCO'
                elif isinstance(eval_dataset, paddlex.datasets.VOCDetection):
                    metric = 'VOC'
                else:
                    raise Exception(
                        "eval_dataset should be datasets.VOCDetection or datasets.COCODetection."
                    )
        assert metric in ['COCO', 'VOC'], "Metric only support 'VOC' or 'COCO'"
        if batch_size > 1:
            batch_size = 1
            logging.warning(
                "Faster RCNN supports batch_size=1 only during evaluating, so batch_size is forced to be set to 1."
            )
        dataset = eval_dataset.generator(
            batch_size=batch_size, drop_last=False)

        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        results = list()
        logging.info(
            "Start to evaluating(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(enumerate(dataset()), total=total_steps):
            images = np.array([d[0] for d in data]).astype('float32')
            im_infos = np.array([d[1] for d in data]).astype('float32')
            im_shapes = np.array([d[3] for d in data]).astype('float32')
            feed_data = {
                'image': images,
                'im_info': im_infos,
                'im_shape': im_shapes,
            }
            outputs = self.exe.run(self.test_prog,
                                   feed=[feed_data],
                                   fetch_list=list(self.test_outputs.values()),
                                   return_numpy=False)
            res = {
                'bbox': (np.array(outputs[0]),
                         outputs[0].recursive_sequence_lengths())
            }
            res_im_id = [d[2] for d in data]
            res['im_info'] = (im_infos, [])
            res['im_shape'] = (im_shapes, [])
            res['im_id'] = (np.array(res_im_id), [])
            if metric == 'VOC':
                res_gt_box = []
                res_gt_label = []
                res_is_difficult = []
                for d in data:
                    res_gt_box.extend(d[4])
                    res_gt_label.extend(d[5])
                    res_is_difficult.extend(d[6])
                res_gt_box_lod = [d[4].shape[0] for d in data]
                res_gt_label_lod = [d[5].shape[0] for d in data]
                res_is_difficult_lod = [d[6].shape[0] for d in data]
                res['gt_box'] = (np.array(res_gt_box), [res_gt_box_lod])
                res['gt_label'] = (np.array(res_gt_label), [res_gt_label_lod])
                res['is_difficult'] = (np.array(res_is_difficult),
                                       [res_is_difficult_lod])
            results.append(res)
            logging.debug("[EVAL] Epoch={}, Step={}/{}".format(epoch_id, step +
                                                               1, total_steps))
        box_ap_stats, eval_details = eval_results(
            results, metric, eval_dataset.coco_gt, with_background=True)
        metrics = OrderedDict(
            zip(['bbox_mmap'
                 if metric == 'COCO' else 'bbox_map'], box_ap_stats))
        if return_details:
            return metrics, eval_details
        return metrics

    def predict(self, img_file, transforms=None):
        """预测。

        Args:
            img_file (str): 预测图像路径。
            transforms (paddlex.det.transforms): 数据预处理操作。

        Returns:
            list: 预测结果列表，每个预测结果由预测框类别标签、
              预测框类别名称、预测框坐标(坐标格式为[xmin, ymin, w, h]）、
              预测框得分组成。
        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is not None:
            self.arrange_transforms(transforms=transforms, mode='test')
            im, im_resize_info, im_shape = transforms(img_file)
        else:
            self.arrange_transforms(
                transforms=self.test_transforms, mode='test')
            im, im_resize_info, im_shape = self.test_transforms(img_file)
        im = np.expand_dims(im, axis=0)
        im_resize_info = np.expand_dims(im_resize_info, axis=0)
        im_shape = np.expand_dims(im_shape, axis=0)
        outputs = self.exe.run(self.test_prog,
                               feed={
                                   'image': im,
                                   'im_info': im_resize_info,
                                   'im_shape': im_shape
                               },
                               fetch_list=list(self.test_outputs.values()),
                               return_numpy=False,
                               use_program_cache=True)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(list(self.test_outputs.keys()), outputs)
        }
        res['im_id'] = (np.array([[0]]).astype('int32'), [])
        clsid2catid = dict({i: i for i in range(self.num_classes)})
        xywh_results = bbox2out([res], clsid2catid)
        results = list()
        for xywh_res in xywh_results:
            del xywh_res['image_id']
            xywh_res['category'] = self.labels[xywh_res['category_id']]
            results.append(xywh_res)
        return results
