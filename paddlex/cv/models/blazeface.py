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
import os.path as osp
import numpy as np
import paddle.fluid as fluid
import paddlex.utils.logging as logging
import paddlex
from .base import BaseAPI
from collections import OrderedDict
import copy

class BlazeFace(BaseAPI):
    """构建BlazeFace，并实现其训练、评估、预测和模型导出。

    Args:
        num_classes (int): 类别数。默认为2。
        backbone (str): YOLOv3的backbone网络，取值范围为['BlazeNet']。默认为'BlazeNet'。
        nms_iou_threshold (float): 进行NMS时，用于剔除检测框IOU的阈值。默认为0.3。
        nms_topk (int): 进行NMS时，根据置信度保留的最大检测框数。默认为5000。
        nms_keep_topk (int): 进行NMS后，每个图像要保留的总检测框数。默认为750。
        nms_score_threshold (float): 检测框的置信度得分阈值，置信度得分低于阈值的框应该被忽略。默认为0.01。
        min_sizes (list): 候选框最小size组成的列表，当use_density_prior_box为False时使用。
        densities (list): 生成候选框的密度，当use_density_prior_box为True时使用。
        use_density_prior_box (bool): 是否使用密度方式获取候选框。默认为False。
    """
    def __init__(self,
                 num_classes=2,
                 backbone='BlazeNet',
                 nms_iou_threshold=0.3,
                 nms_topk=5000,
                 nms_keep_topk=750,
                 nms_score_threshold=0.01,
                 min_sizes=[[16.,24.], [32., 48., 64., 80., 96., 128.]],
                 densities=[[2, 2], [2, 1, 1, 1, 1, 1]],
                 use_density_prior_box=False):
        self.init_params = locals()
        super(BlazeFace, self).__init__('detector')
        backbones = [
            'BlazeNet'
        ]
        assert backbone in backbones, "backbone should be one of {}".format(
            backbones)
        self.backbone = backbone
        self.num_classes = num_classes
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_topk = nms_topk
        self.nms_keep_topk = nms_keep_topk
        self.nms_score_threshold = nms_score_threshold
        self.min_sizes = min_sizes
        self.densities = densities
        self.use_density_prior_box = use_density_prior_box
        self.fixed_input_shape = None
        
    def _get_backbone(self, backbone_name):
        if backbone_name == 'BlazeNet':
            backbone = paddlex.cv.nets.BlazeNet()
        return backbone
    
    def build_net(self, mode='train'):
        model = paddlex.cv.nets.detection.BlazeFace(
            backbone=self._get_backbone(self.backbone),
            min_sizes=self.min_sizes,
            num_classes=self.num_classes,
            use_density_prior_box=self.use_density_prior_box,
            densities=self.densities,
            nms_threshold=self.nms_iou_threshold,
            nms_topk=self.nms_topk,
            nms_keep_topk=self.nms_score_threshold,
            score_threshold=self.nms_score_threshold,
            fixed_input_shape=self.fixed_input_shape)
        inputs = model.generate_inputs()
        model_out = model.build_net(inputs)
        outputs = OrderedDict([('bbox', model_out)])
        if mode == 'train':
            self.optimizer.minimize(model_out)
            outputs = OrderedDict([('loss', model_out)])
        return inputs, outputs
    
    def default_optimizer(self, learning_rate,
                          lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [(lr_decay_gamma**i) * learning_rate
                  for i in range(len(lr_decay_epochs) + 1)]
        lr_decay = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
        optimizer = fluid.optimizer.RMSPropOptimizer(
            learning_rate=lr_decay,
            momentum=0.0,
            regularization=fluid.regularizer.L2DecayRegularizer(5e-04))
        return optimizer
    
    def train(self,
          num_epochs,
          train_dataset,
          train_batch_size=2,
          eval_dataset=None,
          save_interval_epochs=1,
          log_interval_steps=20,
          save_dir='output',
          pretrain_weights=None,
          optimizer=None,
          learning_rate=0.0025,
          lr_decay_epochs=[597, 746],
          lr_decay_gamma=0.1,
          metric='COCO',
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
            pretrain_weights (str): 若指定为路径时，则加载路径下预训练模型；若为None，则不使用预训练模型。默认为None。
            optimizer (paddle.fluid.optimizer): 优化器。当该参数为None时，使用默认优化器：
                fluid.layers.piecewise_decay衰减策略，fluid.optimizer.Momentum优化方法。
            learning_rate (float): 默认优化器的初始学习率。默认为0.001。
            lr_decay_epochs (list): 默认优化器的学习率衰减轮数。默认为[597, 746]。
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
            if isinstances(train_dataset, paddlex.datasets.WIDERFACEDetection):
                metric = 'WIDERFACE'
            elif isinstance(train_dataset, paddlex.datasets.CocoDetection):
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
                learning_rate, lr_decay_epochs,
                lr_decay_gamma, num_steps_each_epoch)
        self.optimizer = optimizer
        # 构建训练、验证、测试网络
        self.build_program()
        self.net_initialize(
            startup_prog=fluid.default_startup_program(),
            pretrain_weights=pretrain_weights,
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
        self.arrange_transforms(transforms=eval_dataset.transforms, mode='eval')
        if metric is None:
            if hasattr(self, 'metric') and self.metric is not None:
                metric = self.metric
            else:
                if isinstance(eval_dataset, paddlex.datasets.CocoDetection):
                    metric = 'COCO'
                elif isinstance(eval_dataset, paddlex.datasets.VOCDetection):
                    metric = 'VOC'
                elif isinstances(train_dataset, paddlex.datasets.WIDERFACEDetection):
                    metric = 'WIDERFACE'
                    logging.info("The metric of WIDERFACE is not supported. This will be implemented soon. " \
                                 + "Now only support 'VOC' or 'COCO'")
                    exit(0)
                else:
                    raise Exception(
                        "eval_dataset should be datasets.VOCDetection or datasets.COCODetection."
                    )
            
        assert metric in ['COCO', 'VOC'], "Metric only support 'VOC' or 'COCO'"
        dataset = eval_dataset.generator(batch_size=batch_size, drop_last=False)

        total_steps = math.ceil(eval_dataset.num_samples * 1.0 / batch_size)
        results = list()
        logging.info("Start to evaluating(total_samples={}, total_steps={})...".
                     format(eval_dataset.num_samples, total_steps))
        for step, data in tqdm.tqdm(enumerate(dataset()), total=total_steps):
            images = np.array([d[0] for d in data]).astype('float32')
            feed_data = {
                'image': images,
            }
            outputs = self.exe.run(self.test_prog,
                                   feed=[feed_data],
                                   fetch_list=list(self.test_outputs.values()),
                                   return_numpy=False)
            res = {
                'bbox': (np.array(outputs[0]),
                         outputs[0].recursive_sequence_lengths())
            }
            res_im_id = [d[4] for d in data]
            res['im_id'] = (np.array(res_im_id), [])
            if metric == 'VOC':
                res_gt_box = []
                res_gt_label = []
                res_is_difficult = []
                for d in data:
                    res_gt_box.extend(d[1])
                    res_gt_label.extend(d[2])
                    res_is_difficult.extend(d[3])
                res_gt_box_lod = [d[1].shape[0] for d in data]
                res_gt_label_lod = [d[2].shape[0] for d in data]
                res_is_difficult_lod = [d[3].shape[0] for d in data]
                res['gt_box'] = (np.array(res_gt_box), [res_gt_box_lod])
                res['gt_label'] = (np.array(res_gt_label), [res_gt_label_lod])
                res['is_difficult'] = (np.array(res_is_difficult),
                                       [res_is_difficult_lod])
            results.append(res)
            logging.debug("[EVAL] Epoch={}, Step={}/{}".format(epoch_id, step +
                                                               1, total_steps))
        box_ap_stats, eval_details = eval_results(
            results, metric, eval_dataset.coco_gt, with_background=True, is_bbox_normalized=True)
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
            im, im_shape = self.test_transforms(img_file)
        im = np.expand_dims(im, axis=0)
        im_shape = np.expand_dims(im_shape, axis=0)
        outputs = self.exe.run(self.test_prog,
                               feed={
                                   'image': im,
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
        xywh_results = bbox2out([res], clsid2catid, is_bbox_normalized=True)
        results = list()
        for xywh_res in xywh_results:
            del xywh_res['image_id']
            xywh_res['category'] = self.labels[xywh_res['category_id']]
            results.append(xywh_res)
        return results
    
    
        
    