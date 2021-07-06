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

import math
import os.path as osp
import numpy as np
from collections import OrderedDict
import paddle
import paddle.nn.functional as F
from paddle.static import InputSpec
import paddlex.paddleseg as paddleseg
import paddlex
from paddlex.cv.transforms import arrange_transforms
from paddlex.utils import get_single_card_bs, DisablePrint
import paddlex.utils.logging as logging
from .base import BaseModel
from .utils import seg_metrics as metrics
from paddlex.utils.checkpoint import seg_pretrain_weights_dict
from paddlex.cv.transforms import Decode, Resize

__all__ = ["UNet", "DeepLabV3P", "FastSCNN", "HRNet", "BiSeNetV2"]


class BaseSegmenter(BaseModel):
    def __init__(self,
                 model_name,
                 num_classes=2,
                 use_mixed_loss=False,
                 **params):
        self.init_params = locals()
        super(BaseSegmenter, self).__init__('segmenter')
        if not hasattr(paddleseg.models, model_name):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_mixed_loss = use_mixed_loss
        self.losses = None
        self.labels = None
        self.net = self.build_net(**params)
        self.find_unused_parameters = True

    def build_net(self, **params):
        # TODO: when using paddle.utils.unique_name.guard,
        # DeepLabv3p and HRNet will raise a error
        net = paddleseg.models.__dict__[self.model_name](
            num_classes=self.num_classes, **params)
        return net

    def _fix_transforms_shape(self, image_shape):
        if hasattr(self, 'test_transforms'):
            if self.test_transforms is not None:
                has_resize_op = False
                resize_op_idx = -1
                normalize_op_idx = len(self.test_transforms.transforms)
                for idx, op in enumerate(self.test_transforms.transforms):
                    name = op.__class__.__name__
                    if name == 'Normalize':
                        normalize_op_idx = idx
                    if 'Resize' in name:
                        has_resize_op = True
                        resize_op_idx = idx

                if not has_resize_op:
                    self.test_transforms.transforms.insert(
                        normalize_op_idx, Resize(target_size=image_shape))
                else:
                    self.test_transforms.transforms[resize_op_idx] = Resize(
                        target_size=image_shape)

    def _get_test_inputs(self, image_shape):
        if image_shape is not None:
            if len(image_shape) == 2:
                image_shape = [1, 3] + image_shape
            self._fix_transforms_shape(image_shape[-2:])
        else:
            image_shape = [None, 3, -1, -1]
        self.fixed_input_shape = image_shape
        input_spec = [
            InputSpec(
                shape=image_shape, name='image', dtype='float32')
        ]
        return input_spec

    def run(self, net, inputs, mode):
        net_out = net(inputs[0])
        logit = net_out[0]
        outputs = OrderedDict()
        if mode == 'test':
            origin_shape = inputs[1]
            score_map = self._postprocess(
                logit, origin_shape, transforms=inputs[2])
            label_map = paddle.argmax(
                score_map, axis=1, keepdim=True, dtype='int32')
            score_map = paddle.max(score_map, axis=1, keepdim=True)
            score_map = paddle.squeeze(score_map)
            label_map = paddle.squeeze(label_map)
            outputs = {'label_map': label_map, 'score_map': score_map}
        if mode == 'eval':
            pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
            label = inputs[1]
            origin_shape = [label.shape[-2:]]
            # TODO: 替换cv2后postprocess移出run
            pred = self._postprocess(pred, origin_shape, transforms=inputs[2])
            intersect_area, pred_area, label_area = paddleseg.utils.metrics.calculate_area(
                pred, label, self.num_classes)
            outputs['intersect_area'] = intersect_area
            outputs['pred_area'] = pred_area
            outputs['label_area'] = label_area
            outputs['conf_mat'] = metrics.confusion_matrix(pred, label,
                                                           self.num_classes)
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
                    paddleseg.models.CrossEntropyLoss(),
                    paddleseg.models.LovaszSoftmaxLoss()
                ]
                coef = [.8, .2]
                loss_type = [
                    paddleseg.models.MixedLoss(
                        losses=losses, coef=coef),
                ]
            else:
                loss_type = [paddleseg.models.CrossEntropyLoss()]
        else:
            losses, coef = list(zip(*self.use_mixed_loss))
            if not set(losses).issubset(
                ['CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss']):
                raise ValueError(
                    "Only 'CrossEntropyLoss', 'DiceLoss', 'LovaszSoftmaxLoss' are supported."
                )
            losses = [getattr(paddleseg.models, loss)() for loss in losses]
            loss_type = [
                paddleseg.models.MixedLoss(
                    losses=losses, coef=list(coef))
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
              early_stop_patience=5,
              use_vdl=True,
              resume_checkpoint=None):
        """
        Train the model.
        Args:
            num_epochs(int): The number of epochs.
            train_dataset(paddlex.dataset): Training dataset.
            train_batch_size(int, optional): Total batch size among all cards used in training. Defaults to 2.
            eval_dataset(paddlex.dataset, optional):
                Evaluation dataset. If None, the model will not be evaluated furing training process. Defaults to None.
            optimizer(paddle.optimizer.Optimizer or None, optional):
                Optimizer used in training. If None, a default optimizer is used. Defaults to None.
            save_interval_epochs(int, optional): Epoch interval for saving the model. Defaults to 1.
            log_interval_steps(int, optional): Step interval for printing training information. Defaults to 10.
            save_dir(str, optional): Directory to save the model. Defaults to 'output'.
            pretrain_weights(str or None, optional):
                None or name/path of pretrained weights. If None, no pretrained weights will be loaded. Defaults to 'CITYSCAPES'.
            learning_rate(float, optional): Learning rate for training. Defaults to .025.
            lr_decay_power(float, optional): Learning decay power. Defaults to .9.
            early_stop(bool, optional): Whether to adopt early stop strategy. Defaults to False.
            early_stop_patience(int, optional): Early stop patience. Defaults to 5.
            use_vdl(bool, optional): Whether to use VisualDL to monitor the training process. Defaults to True.
            resume_checkpoint(str or None, optional): The path of the checkpoint to resume training from.
                If None, no training checkpoint will be resumed. At most one of `resume_checkpoint` and
                `pretrain_weights` can be set simultaneously. Defaults to None.

        """
        if pretrain_weights is not None and resume_checkpoint is not None:
            logging.error(
                "pretrain_weights and resume_checkpoint cannot be set simultaneously.",
                exit=True)
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
            if pretrain_weights not in seg_pretrain_weights_dict[
                    self.model_name]:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                logging.warning("Pretrain_weights is forcibly set to '{}'. "
                                "If don't want to use pretrain weights, "
                                "set pretrain_weights to be None.".format(
                                    seg_pretrain_weights_dict[self.model_name][
                                        0]))
                pretrain_weights = seg_pretrain_weights_dict[self.model_name][
                    0]
        elif pretrain_weights is not None and osp.exists(pretrain_weights):
            if osp.splitext(pretrain_weights)[-1] != '.pdparams':
                logging.error(
                    "Invalid pretrain weights. Please specify a '.pdparams' file.",
                    exit=True)
        pretrained_dir = osp.join(save_dir, 'pretrain')
        is_backbone_weights = pretrain_weights == 'IMAGENET'
        self.net_initialize(
            pretrain_weights=pretrain_weights,
            save_dir=pretrained_dir,
            resume_checkpoint=resume_checkpoint,
            is_backbone_weights=is_backbone_weights)

        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl)

    def quant_aware_train(self,
                          num_epochs,
                          train_dataset,
                          train_batch_size=2,
                          eval_dataset=None,
                          optimizer=None,
                          save_interval_epochs=1,
                          log_interval_steps=2,
                          save_dir='output',
                          learning_rate=0.0001,
                          lr_decay_power=0.9,
                          early_stop=False,
                          early_stop_patience=5,
                          use_vdl=True,
                          resume_checkpoint=None,
                          quant_config=None):
        """
        Quantization-aware training.
        Args:
            num_epochs(int): The number of epochs.
            train_dataset(paddlex.dataset): Training dataset.
            train_batch_size(int, optional): Total batch size among all cards used in training. Defaults to 2.
            eval_dataset(paddlex.dataset, optional):
                Evaluation dataset. If None, the model will not be evaluated furing training process. Defaults to None.
            optimizer(paddle.optimizer.Optimizer or None, optional):
                Optimizer used in training. If None, a default optimizer is used. Defaults to None.
            save_interval_epochs(int, optional): Epoch interval for saving the model. Defaults to 1.
            log_interval_steps(int, optional): Step interval for printing training information. Defaults to 10.
            save_dir(str, optional): Directory to save the model. Defaults to 'output'.
            learning_rate(float, optional): Learning rate for training. Defaults to .025.
            lr_decay_power(float, optional): Learning decay power. Defaults to .9.
            early_stop(bool, optional): Whether to adopt early stop strategy. Defaults to False.
            early_stop_patience(int, optional): Early stop patience. Defaults to 5.
            use_vdl(bool, optional): Whether to use VisualDL to monitor the training process. Defaults to True.
            quant_config(dict or None, optional): Quantization configuration. If None, a default rule of thumb
                configuration will be used. Defaults to None.
            resume_checkpoint(str or None, optional): The path of the checkpoint to resume quantization-aware training
                from. If None, no training checkpoint will be resumed. Defaults to None.

        """
        self._prepare_qat(quant_config)
        self.train(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            optimizer=optimizer,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=None,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl,
            resume_checkpoint=resume_checkpoint)

    def evaluate(self, eval_dataset, batch_size=1, return_details=False):
        """
        Evaluate the model.
        Args:
            eval_dataset(paddlex.dataset): Evaluation dataset.
            batch_size(int, optional): Total batch size among all cards used for evaluation. Defaults to 1.
            return_details(bool, optional): Whether to return evaluation details. Defaults to False.

        Returns:
            collections.OrderedDict with key-value pairs:
                {"miou": `mean intersection over union`,
                 "category_iou": `category-wise mean intersection over union`,
                 "oacc": `overall accuracy`,
                 "category_acc": `category-wise accuracy`,
                 "kappa": ` kappa coefficient`,
                 "category_F1-score": `F1 score`}.

        """
        arrange_transforms(
            model_type=self.model_type,
            transforms=eval_dataset.transforms,
            mode='eval')

        self.net.eval()
        nranks = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
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
                "Segmenter only supports batch_size=1 for each gpu/cpu card " \
                "during evaluation, so batch_size " \
                "is forcibly set to {}.".format(batch_size))
        self.eval_data_loader = self.build_data_loader(
            eval_dataset, batch_size=batch_size, mode='eval')

        intersect_area_all = 0
        pred_area_all = 0
        label_area_all = 0
        conf_mat_all = []
        logging.info(
            "Start to evaluate(total_samples={}, total_steps={})...".format(
                eval_dataset.num_samples,
                math.ceil(eval_dataset.num_samples * 1.0 / batch_size)))
        with paddle.no_grad():
            for step, data in enumerate(self.eval_data_loader):
                data.append(eval_dataset.transforms.transforms)
                outputs = self.run(self.net, data, 'eval')
                pred_area = outputs['pred_area']
                label_area = outputs['label_area']
                intersect_area = outputs['intersect_area']
                conf_mat = outputs['conf_mat']

                # Gather from all ranks
                if nranks > 1:
                    intersect_area_list = []
                    pred_area_list = []
                    label_area_list = []
                    conf_mat_list = []
                    paddle.distributed.all_gather(intersect_area_list,
                                                  intersect_area)
                    paddle.distributed.all_gather(pred_area_list, pred_area)
                    paddle.distributed.all_gather(label_area_list, label_area)
                    paddle.distributed.all_gather(conf_mat_list, conf_mat)

                    # Some image has been evaluated and should be eliminated in last iter
                    if (step + 1) * nranks > len(eval_dataset):
                        valid = len(eval_dataset) - step * nranks
                        intersect_area_list = intersect_area_list[:valid]
                        pred_area_list = pred_area_list[:valid]
                        label_area_list = label_area_list[:valid]
                        conf_mat_list = conf_mat_list[:valid]

                    intersect_area_all += sum(intersect_area_list)
                    pred_area_all += sum(pred_area_list)
                    label_area_all += sum(label_area_list)
                    conf_mat_all.extend(conf_mat_list)

                else:
                    intersect_area_all = intersect_area_all + intersect_area
                    pred_area_all = pred_area_all + pred_area
                    label_area_all = label_area_all + label_area
                    conf_mat_all.append(conf_mat)
        class_iou, miou = paddleseg.utils.metrics.mean_iou(
            intersect_area_all, pred_area_all, label_area_all)
        # TODO 确认是按oacc还是macc
        class_acc, oacc = paddleseg.utils.metrics.accuracy(intersect_area_all,
                                                           pred_area_all)
        kappa = paddleseg.utils.metrics.kappa(intersect_area_all,
                                              pred_area_all, label_area_all)
        category_f1score = metrics.f1_score(intersect_area_all, pred_area_all,
                                            label_area_all)
        eval_metrics = OrderedDict(
            zip([
                'miou', 'category_iou', 'oacc', 'category_acc', 'kappa',
                'category_F1-score'
            ], [miou, class_iou, oacc, class_acc, kappa, category_f1score]))

        if return_details:
            conf_mat = sum(conf_mat_all)
            eval_details = {'confusion_matrix': conf_mat.tolist()}
            return eval_metrics, eval_details
        return eval_metrics

    def predict(self, img_file, transforms=None):
        """
        Do inference.
        Args:
            Args:
            img_file(List[np.ndarray or str], str or np.ndarray): img_file(list or str or np.array)：
                Image path or decoded image data in a BGR format, which also could constitute a list,
                meaning all images to be predicted as a mini-batch.
            transforms(paddlex.transforms.Compose or None, optional):
                Transforms for inputs. If None, the transforms for evaluation process will be used. Defaults to None.

        Returns:
            If img_file is a string or np.array, the result is a dict with key-value pairs:
            {"label map": `label map`, "score_map": `score map`}.
            If img_file is a list, the result is a list composed of dicts with the corresponding fields:
            label_map(np.ndarray): the predicted label map
            score_map(np.ndarray): the prediction score map

        """
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file
        batch_im, batch_origin_shape = self._preprocess(images, transforms,
                                                        self.model_type)
        self.net.eval()
        data = (batch_im, batch_origin_shape, transforms.transforms)
        outputs = self.run(self.net, data, 'test')
        label_map = outputs['label_map']
        label_map = label_map.numpy().astype('uint8')
        score_map = outputs['score_map']
        score_map = score_map.numpy().astype('float32')
        if isinstance(img_file, list) and len(img_file) > 1:
            prediction = [{
                'label_map': l,
                'score_map': s
            } for l, s in zip(label_map, score_map)]
        elif isinstance(img_file, list):
            prediction = [{'label_map': label_map, 'score_map': score_map}]
        else:
            prediction = {'label_map': label_map, 'score_map': score_map}
        return prediction

    def _preprocess(self, images, transforms, model_type):
        arrange_transforms(
            model_type=model_type, transforms=transforms, mode='test')
        batch_im = list()
        batch_ori_shape = list()
        for im in images:
            sample = {'image': im}
            if isinstance(sample['image'], str):
                sample = Decode(to_rgb=False)(sample)
            ori_shape = sample['image'].shape[:2]
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
                    h, w = op.target_size
                if op.__class__.__name__ in ['Padding']:
                    restore_list.append(('padding', (h, w)))
                    h, w = op.target_size
            batch_restore_list.append(restore_list)
        return batch_restore_list

    def _postprocess(self, batch_pred, batch_origin_shape, transforms):
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
        with DisablePrint():
            backbone = getattr(paddleseg.models, backbone)(
                output_stride=output_stride)
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
        with DisablePrint():
            backbone = getattr(paddleseg.models, self.backbone_name)(
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
