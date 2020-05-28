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
import os
import sys
import numpy as np
import time
import math
import yaml
import copy
import json
import functools
import paddlex.utils.logging as logging
from paddlex.utils import seconds_to_hms
from paddlex.utils.utils import EarlyStop
import paddlex
from collections import OrderedDict
from os import path as osp
from paddle.fluid.framework import Program
from .utils.pretrain_weights import get_pretrain_weights


def dict2str(dict_input):
    out = ''
    for k, v in dict_input.items():
        try:
            v = round(float(v), 6)
        except:
            pass
        out = out + '{}={}, '.format(k, v)
    return out.strip(', ')


class BaseAPI:
    def __init__(self, model_type):
        self.model_type = model_type
        # 现有的CV模型都有这个属性，而这个属且也需要在eval时用到
        self.num_classes = None
        self.labels = None
        self.version = paddlex.__version__
        if paddlex.env_info['place'] == 'cpu':
            self.places = fluid.cpu_places()
        else:
            self.places = fluid.cuda_places()
        self.exe = fluid.Executor(self.places[0])
        self.train_prog = None
        self.test_prog = None
        self.parallel_train_prog = None
        self.train_inputs = None
        self.test_inputs = None
        self.train_outputs = None
        self.test_outputs = None
        self.train_data_loader = None
        self.eval_metrics = None
        # 若模型是从inference model加载进来的，无法调用训练接口进行训练
        self.trainable = True
        # 是否使用多卡间同步BatchNorm均值和方差
        self.sync_bn = False
        # 当前模型状态
        self.status = 'Normal'
        # 已完成迭代轮数，为恢复训练时的起始轮数
        self.completed_epochs = 0

    def _get_single_card_bs(self, batch_size):
        if batch_size % len(self.places) == 0:
            return int(batch_size // len(self.places))
        else:
            raise Exception("Please support correct batch_size, \
                            which can be divided by available cards({}) in {}"
                            .format(paddlex.env_info['num'], paddlex.env_info[
                                'place']))

    def build_program(self):
        # 构建训练网络
        self.train_inputs, self.train_outputs = self.build_net(mode='train')
        self.train_prog = fluid.default_main_program()
        startup_prog = fluid.default_startup_program()

        # 构建预测网络
        self.test_prog = fluid.Program()
        with fluid.program_guard(self.test_prog, startup_prog):
            with fluid.unique_name.guard():
                self.test_inputs, self.test_outputs = self.build_net(
                    mode='test')
        self.test_prog = self.test_prog.clone(for_test=True)

    def arrange_transforms(self, transforms, mode='train'):
        # 给transforms添加arrange操作
        if self.model_type == 'classifier':
            arrange_transform = paddlex.cls.transforms.ArrangeClassifier
        elif self.model_type == 'segmenter':
            arrange_transform = paddlex.seg.transforms.ArrangeSegmenter
        elif self.model_type == 'detector':
            arrange_name = 'Arrange{}'.format(self.__class__.__name__)
            arrange_transform = getattr(paddlex.det.transforms, arrange_name)
        else:
            raise Exception("Unrecognized model type: {}".format(
                self.model_type))
        if type(transforms.transforms[-1]).__name__.startswith('Arrange'):
            transforms.transforms[-1] = arrange_transform(mode=mode)
        else:
            transforms.transforms.append(arrange_transform(mode=mode))

    def build_train_data_loader(self, dataset, batch_size):
        # 初始化data_loader
        if self.train_data_loader is None:
            self.train_data_loader = fluid.io.DataLoader.from_generator(
                feed_list=list(self.train_inputs.values()),
                capacity=64,
                use_double_buffer=True,
                iterable=True)
        batch_size_each_gpu = self._get_single_card_bs(batch_size)
        generator = dataset.generator(
            batch_size=batch_size_each_gpu, drop_last=True)
        self.train_data_loader.set_sample_list_generator(
            dataset.generator(batch_size=batch_size_each_gpu),
            places=self.places)

    def export_quant_model(self,
                           dataset,
                           save_dir,
                           batch_size=1,
                           batch_num=10,
                           cache_dir="./temp"):
        self.arrange_transforms(transforms=dataset.transforms, mode='quant')
        dataset.num_samples = batch_size * batch_num
        try:
            from .slim.post_quantization import PaddleXPostTrainingQuantization
            PaddleXPostTrainingQuantization._collect_target_varnames
        except:
            raise Exception(
                "Model Quantization is not available, try to upgrade your paddlepaddle>=1.8.0"
            )
        is_use_cache_file = True
        if cache_dir is None:
            is_use_cache_file = False
        post_training_quantization = PaddleXPostTrainingQuantization(
            executor=self.exe,
            dataset=dataset,
            program=self.test_prog,
            inputs=self.test_inputs,
            outputs=self.test_outputs,
            batch_size=batch_size,
            batch_nums=batch_num,
            scope=None,
            algo='KL',
            quantizable_op_type=["conv2d", "depthwise_conv2d", "mul"],
            is_full_quantize=False,
            is_use_cache_file=is_use_cache_file,
            cache_dir=cache_dir)
        post_training_quantization.quantize()
        post_training_quantization.save_quantized_model(save_dir)
        model_info = self.get_model_info()
        model_info['status'] = 'Quant'

        # 保存模型输出的变量描述
        model_info['_ModelInputsOutputs'] = dict()
        model_info['_ModelInputsOutputs']['test_inputs'] = [
            [k, v.name] for k, v in self.test_inputs.items()
        ]
        model_info['_ModelInputsOutputs']['test_outputs'] = [
            [k, v.name] for k, v in self.test_outputs.items()
        ]

        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

    def net_initialize(self,
                       startup_prog=None,
                       pretrain_weights=None,
                       fuse_bn=False,
                       save_dir='.',
                       sensitivities_file=None,
                       eval_metric_loss=0.05,
                       resume_checkpoint=None):
        if not resume_checkpoint:
            pretrain_dir = osp.join(save_dir, 'pretrain')
            if not os.path.isdir(pretrain_dir):
                if os.path.exists(pretrain_dir):
                    os.remove(pretrain_dir)
                os.makedirs(pretrain_dir)
            if hasattr(self, 'backbone'):
                backbone = self.backbone
            else:
                backbone = self.__class__.__name__
                if backbone == "HRNet":
                    backbone = backbone + "_W{}".format(self.width)
            pretrain_weights = get_pretrain_weights(
                pretrain_weights, self.model_type, backbone, pretrain_dir)
        if startup_prog is None:
            startup_prog = fluid.default_startup_program()
        self.exe.run(startup_prog)
        if resume_checkpoint:
            logging.info(
                "Resume checkpoint from {}.".format(resume_checkpoint),
                use_color=True)
            paddlex.utils.utils.load_pretrain_weights(
                self.exe, self.train_prog, resume_checkpoint, resume=True)
            if not osp.exists(osp.join(resume_checkpoint, "model.yml")):
                raise Exception("There's not model.yml in {}".format(
                    resume_checkpoint))
            with open(osp.join(resume_checkpoint, "model.yml")) as f:
                info = yaml.load(f.read(), Loader=yaml.Loader)
                self.completed_epochs = info['completed_epochs']
        elif pretrain_weights is not None:
            logging.info(
                "Load pretrain weights from {}.".format(pretrain_weights),
                use_color=True)
            paddlex.utils.utils.load_pretrain_weights(
                self.exe, self.train_prog, pretrain_weights, fuse_bn)
        # 进行裁剪
        if sensitivities_file is not None:
            import paddleslim
            from .slim.prune_config import get_sensitivities
            sensitivities_file = get_sensitivities(sensitivities_file, self,
                                                   save_dir)
            from .slim.prune import get_params_ratios, prune_program
            logging.info(
                "Start to prune program with eval_metric_loss = {}".format(
                    eval_metric_loss),
                use_color=True)
            origin_flops = paddleslim.analysis.flops(self.test_prog)
            prune_params_ratios = get_params_ratios(
                sensitivities_file, eval_metric_loss=eval_metric_loss)
            prune_program(self, prune_params_ratios)
            current_flops = paddleslim.analysis.flops(self.test_prog)
            remaining_ratio = current_flops / origin_flops
            logging.info(
                "Finish prune program, before FLOPs:{}, after prune FLOPs:{}, remaining ratio:{}"
                .format(origin_flops, current_flops, remaining_ratio),
                use_color=True)
            self.status = 'Prune'

    def get_model_info(self):
        info = dict()
        info['version'] = paddlex.__version__
        info['Model'] = self.__class__.__name__
        info['_Attributes'] = {'model_type': self.model_type}
        if 'self' in self.init_params:
            del self.init_params['self']
        if '__class__' in self.init_params:
            del self.init_params['__class__']
        if 'model_name' in self.init_params:
            del self.init_params['model_name']

        info['_init_params'] = self.init_params

        info['_Attributes']['num_classes'] = self.num_classes
        info['_Attributes']['labels'] = self.labels
        try:
            primary_metric_key = list(self.eval_metrics.keys())[0]
            primary_metric_value = float(self.eval_metrics[primary_metric_key])
            info['_Attributes']['eval_metrics'] = {
                primary_metric_key: primary_metric_value
            }
        except:
            pass

        if hasattr(self, 'test_transforms'):
            if hasattr(self.test_transforms, 'to_rgb'):
                if self.test_transforms.to_rgb:
                    info['TransformsMode'] = 'RGB'
                else:
                    info['TransformsMode'] = 'BGR'

            if self.test_transforms is not None:
                info['Transforms'] = list()
                for op in self.test_transforms.transforms:
                    name = op.__class__.__name__
                    attr = op.__dict__
                    info['Transforms'].append({name: attr})
        info['completed_epochs'] = self.completed_epochs
        return info

    def save_model(self, save_dir):
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        if self.train_prog is not None:
            fluid.save(self.train_prog, osp.join(save_dir, 'model'))
        else:
            fluid.save(self.test_prog, osp.join(save_dir, 'model'))
        model_info = self.get_model_info()
        model_info['status'] = self.status
        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)
        # 评估结果保存
        if hasattr(self, 'eval_details'):
            with open(osp.join(save_dir, 'eval_details.json'), 'w') as f:
                json.dump(self.eval_details, f)

        if self.status == 'Prune':
            # 保存裁剪的shape
            shapes = {}
            for block in self.train_prog.blocks:
                for param in block.all_parameters():
                    pd_var = fluid.global_scope().find_var(param.name)
                    pd_param = pd_var.get_tensor()
                    shapes[param.name] = np.array(pd_param).shape
            with open(
                    osp.join(save_dir, 'prune.yml'), encoding='utf-8',
                    mode='w') as f:
                yaml.dump(shapes, f)

        # 模型保存成功的标志
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model saved in {}.".format(save_dir))

    def export_inference_model(self, save_dir):
        test_input_names = [
            var.name for var in list(self.test_inputs.values())
        ]
        test_outputs = list(self.test_outputs.values())
        if self.__class__.__name__ == 'MaskRCNN':
            from paddlex.utils.save import save_mask_inference_model
            save_mask_inference_model(
                dirname=save_dir,
                executor=self.exe,
                params_filename='__params__',
                feeded_var_names=test_input_names,
                target_vars=test_outputs,
                main_program=self.test_prog)
        else:
            fluid.io.save_inference_model(
                dirname=save_dir,
                executor=self.exe,
                params_filename='__params__',
                feeded_var_names=test_input_names,
                target_vars=test_outputs,
                main_program=self.test_prog)
        model_info = self.get_model_info()
        model_info['status'] = 'Infer'

        # 保存模型输出的变量描述
        model_info['_ModelInputsOutputs'] = dict()
        model_info['_ModelInputsOutputs']['test_inputs'] = [
            [k, v.name] for k, v in self.test_inputs.items()
        ]
        model_info['_ModelInputsOutputs']['test_outputs'] = [
            [k, v.name] for k, v in self.test_outputs.items()
        ]
        with open(
                osp.join(save_dir, 'model.yml'), encoding='utf-8',
                mode='w') as f:
            yaml.dump(model_info, f)

        # 模型保存成功的标志
        open(osp.join(save_dir, '.success'), 'w').close()
        logging.info("Model for inference deploy saved in {}.".format(
            save_dir))

    def train_loop(self,
                   num_epochs,
                   train_dataset,
                   train_batch_size,
                   eval_dataset=None,
                   save_interval_epochs=1,
                   log_interval_steps=10,
                   save_dir='output',
                   use_vdl=False,
                   early_stop=False,
                   early_stop_patience=5):
        if train_dataset.num_samples < train_batch_size:
            raise Exception(
                'The amount of training datset must be larger than batch size.')
        if not osp.isdir(save_dir):
            if osp.exists(save_dir):
                os.remove(save_dir)
            os.makedirs(save_dir)
        if use_vdl:
            from visualdl import LogWriter
            vdl_logdir = osp.join(save_dir, 'vdl_log')
        # 给transform添加arrange操作
        self.arrange_transforms(
            transforms=train_dataset.transforms, mode='train')
        # 构建train_data_loader
        self.build_train_data_loader(
            dataset=train_dataset, batch_size=train_batch_size)

        if eval_dataset is not None:
            self.eval_transforms = eval_dataset.transforms
            self.test_transforms = copy.deepcopy(eval_dataset.transforms)

        # 获取实时变化的learning rate
        lr = self.optimizer._learning_rate
        if isinstance(lr, fluid.framework.Variable):
            self.train_outputs['lr'] = lr

        # 在多卡上跑训练
        if self.parallel_train_prog is None:
            build_strategy = fluid.compiler.BuildStrategy()
            build_strategy.fuse_all_optimizer_ops = False
            if paddlex.env_info['place'] != 'cpu' and len(self.places) > 1:
                build_strategy.sync_batch_norm = self.sync_bn
            exec_strategy = fluid.ExecutionStrategy()
            exec_strategy.num_iteration_per_drop_scope = 1
            self.parallel_train_prog = fluid.CompiledProgram(
                self.train_prog).with_data_parallel(
                    loss_name=self.train_outputs['loss'].name,
                    build_strategy=build_strategy,
                    exec_strategy=exec_strategy)

        total_num_steps = math.floor(train_dataset.num_samples /
                                     train_batch_size)
        num_steps = 0
        time_stat = list()
        time_train_one_epoch = None
        time_eval_one_epoch = None

        total_num_steps_eval = 0
        # 模型总共的评估次数
        total_eval_times = math.ceil(num_epochs / save_interval_epochs)
        # 检测目前仅支持单卡评估，训练数据batch大小与显卡数量之商为验证数据batch大小。
        eval_batch_size = train_batch_size
        if self.model_type == 'detector':
            eval_batch_size = self._get_single_card_bs(train_batch_size)
        if eval_dataset is not None:
            total_num_steps_eval = math.ceil(eval_dataset.num_samples /
                                             eval_batch_size)

        if use_vdl:
            # VisualDL component
            log_writer = LogWriter(vdl_logdir)

        thresh = 0.0001
        if early_stop:
            earlystop = EarlyStop(early_stop_patience, thresh)
        best_accuracy_key = ""
        best_accuracy = -1.0
        best_model_epoch = -1
        start_epoch = self.completed_epochs
        for i in range(start_epoch, num_epochs):
            records = list()
            step_start_time = time.time()
            epoch_start_time = time.time()
            for step, data in enumerate(self.train_data_loader()):
                outputs = self.exe.run(
                    self.parallel_train_prog,
                    feed=data,
                    fetch_list=list(self.train_outputs.values()))
                outputs_avg = np.mean(np.array(outputs), axis=1)
                records.append(outputs_avg)

                # 训练完成剩余时间预估
                current_time = time.time()
                step_cost_time = current_time - step_start_time
                step_start_time = current_time
                if len(time_stat) < 20:
                    time_stat.append(step_cost_time)
                else:
                    time_stat[num_steps % 20] = step_cost_time

                # 每间隔log_interval_steps，输出loss信息
                num_steps += 1
                if num_steps % log_interval_steps == 0:
                    step_metrics = OrderedDict(
                        zip(list(self.train_outputs.keys()), outputs_avg))

                    if use_vdl:
                        for k, v in step_metrics.items():
                            log_writer.add_scalar(
                                'Metrics/Training(Step): {}'.format(k), v,
                                num_steps)

                    # 估算剩余时间
                    avg_step_time = np.mean(time_stat)
                    if time_train_one_epoch is not None:
                        eta = (num_epochs - i - 1) * time_train_one_epoch + (
                            total_num_steps - step - 1) * avg_step_time
                    else:
                        eta = ((num_epochs - i) * total_num_steps - step - 1
                               ) * avg_step_time
                    if time_eval_one_epoch is not None:
                        eval_eta = (
                            total_eval_times - i // save_interval_epochs
                        ) * time_eval_one_epoch
                    else:
                        eval_eta = (
                            total_eval_times - i // save_interval_epochs
                        ) * total_num_steps_eval * avg_step_time
                    eta_str = seconds_to_hms(eta + eval_eta)

                    logging.info(
                        "[TRAIN] Epoch={}/{}, Step={}/{}, {}, time_each_step={}s, eta={}"
                        .format(i + 1, num_epochs, step + 1, total_num_steps,
                                dict2str(step_metrics),
                                round(avg_step_time, 2), eta_str))
            train_metrics = OrderedDict(
                zip(list(self.train_outputs.keys()), np.mean(
                    records, axis=0)))
            logging.info('[TRAIN] Epoch {} finished, {} .'.format(
                i + 1, dict2str(train_metrics)))
            time_train_one_epoch = time.time() - epoch_start_time
            epoch_start_time = time.time()

            # 每间隔save_interval_epochs, 在验证集上评估和对模型进行保存
            eval_epoch_start_time = time.time()
            if (i + 1) % save_interval_epochs == 0 or i == num_epochs - 1:
                current_save_dir = osp.join(save_dir, "epoch_{}".format(i + 1))
                if not osp.isdir(current_save_dir):
                    os.makedirs(current_save_dir)
                if eval_dataset is not None and eval_dataset.num_samples > 0:
                    self.eval_metrics, self.eval_details = self.evaluate(
                        eval_dataset=eval_dataset,
                        batch_size=eval_batch_size,
                        epoch_id=i + 1,
                        return_details=True)
                    logging.info('[EVAL] Finished, Epoch={}, {} .'.format(
                        i + 1, dict2str(self.eval_metrics)))
                    self.completed_epochs += 1
                    # 保存最优模型
                    best_accuracy_key = list(self.eval_metrics.keys())[0]
                    current_accuracy = self.eval_metrics[best_accuracy_key]
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_model_epoch = i + 1
                        best_model_dir = osp.join(save_dir, "best_model")
                        self.save_model(save_dir=best_model_dir)
                    if use_vdl:
                        for k, v in self.eval_metrics.items():
                            if isinstance(v, list):
                                continue
                            if isinstance(v, np.ndarray):
                                if v.size > 1:
                                    continue
                            log_writer.add_scalar(
                                "Metrics/Eval(Epoch): {}".format(k), v, i + 1)
                self.save_model(save_dir=current_save_dir)
                time_eval_one_epoch = time.time() - eval_epoch_start_time
                eval_epoch_start_time = time.time()
                if best_model_epoch > 0:
                    logging.info(
                        'Current evaluated best model in eval_dataset is epoch_{}, {}={}'
                        .format(best_model_epoch, best_accuracy_key,
                                best_accuracy))
                if eval_dataset is not None and early_stop:
                    if earlystop(current_accuracy):
                        break
