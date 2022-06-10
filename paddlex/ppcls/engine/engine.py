# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random

from paddlex.ppcls.utils.check import check_gpu
from paddlex.ppcls.utils.misc import AverageMeter
from paddlex.ppcls.utils import logger
from paddlex.ppcls.utils.logger import init_logger
from paddlex.ppcls.utils.config import print_config
from paddlex.ppcls.data import build_dataloader
from paddlex.ppcls.arch import build_model, RecModel, DistillationModel
from paddlex.ppcls.arch import apply_to_static
from paddlex.ppcls.loss import build_loss
from paddlex.ppcls.metric import build_metrics
from paddlex.ppcls.optimizer import build_optimizer
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from paddlex.ppcls.utils.save_load import init_model
from paddlex.ppcls.utils import save_load

from paddlex.ppcls.data.utils.get_image_list import get_image_list
from paddlex.ppcls.data.postprocess import build_postprocess
from paddlex.ppcls.data import create_operators
from paddlex.ppcls.engine.train import train_epoch
from paddlex.ppcls.engine import evaluation
from paddlex.ppcls.arch.gears.identity_head import IdentityHead
from paddlex.ppcls.engine.slim import get_pruner, get_quaner


class Engine(object):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode = mode
        self.config = config
        self.eval_mode = self.config["Global"].get("eval_mode",
                                                   "classification")
        if "Head" in self.config["Arch"]:
            self.is_rec = True
        else:
            self.is_rec = False

        # set seed
        seed = self.config["Global"].get("seed", False)
        if seed or seed == 0:
            assert isinstance(seed, int), "The 'seed' must be a integer!"
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # init logger
        self.output_dir = self.config['Global']['output_dir']
        log_file = os.path.join(self.output_dir, self.config["Arch"]["name"],
                                f"{mode}.log")
        init_logger(name='root', log_file=log_file)
        print_config(config)

        # init train_func and eval_func
        assert self.eval_mode in ["classification", "retrieval"], logger.error(
            "Invalid eval mode: {}".format(self.eval_mode))
        self.train_epoch_func = train_epoch
        self.eval_func = getattr(evaluation, self.eval_mode + "_eval")

        self.use_dali = self.config['Global'].get("use_dali", False)

        # for visualdl
        self.vdl_writer = None
        if self.config['Global']['use_visualdl'] and mode == "train":
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)

        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu", "npu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))

        # AMP training
        self.amp = True if "AMP" in self.config else False
        if self.amp and self.config["AMP"] is not None:
            self.scale_loss = self.config["AMP"].get("scale_loss", 1.0)
            self.use_dynamic_loss_scaling = self.config["AMP"].get(
                "use_dynamic_loss_scaling", False)
        else:
            self.scale_loss = 1.0
            self.use_dynamic_loss_scaling = False
        if self.amp:
            AMP_RELATED_FLAGS_SETTING = {
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
                'FLAGS_max_inplace_grad_add': 8,
            }
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)

        #TODO(gaotingquan): support rec
        class_num = config["Arch"].get("class_num", None)
        self.config["DataLoader"].update({"class_num": class_num})
        # build dataloader
        if self.mode == 'train':
            self.train_dataloader = build_dataloader(
                self.config["DataLoader"], "Train", self.device, self.use_dali)
        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            if self.eval_mode == "classification":
                self.eval_dataloader = build_dataloader(
                    self.config["DataLoader"], "Eval", self.device,
                    self.use_dali)
            elif self.eval_mode == "retrieval":
                self.gallery_query_dataloader = None
                if len(self.config["DataLoader"]["Eval"].keys()) == 1:
                    key = list(self.config["DataLoader"]["Eval"].keys())[0]
                    self.gallery_query_dataloader = build_dataloader(
                        self.config["DataLoader"]["Eval"], key, self.device,
                        self.use_dali)
                else:
                    self.gallery_dataloader = build_dataloader(
                        self.config["DataLoader"]["Eval"], "Gallery",
                        self.device, self.use_dali)
                    self.query_dataloader = build_dataloader(
                        self.config["DataLoader"]["Eval"], "Query",
                        self.device, self.use_dali)

        # build loss
        if self.mode == "train":
            loss_info = self.config["Loss"]["Train"]
            self.train_loss_func = build_loss(loss_info)
        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            loss_config = self.config.get("Loss", None)
            if loss_config is not None:
                loss_config = loss_config.get("Eval")
                if loss_config is not None:
                    self.eval_loss_func = build_loss(loss_config)
                else:
                    self.eval_loss_func = None
            else:
                self.eval_loss_func = None

        # build metric
        if self.mode == 'train':
            metric_config = self.config.get("Metric")
            if metric_config is not None:
                metric_config = metric_config.get("Train")
                if metric_config is not None:
                    self.train_metric_func = build_metrics(metric_config)
                else:
                    self.train_metric_func = None
        else:
            self.train_metric_func = None

        if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            metric_config = self.config.get("Metric")
            if self.eval_mode == "classification":
                if metric_config is not None:
                    metric_config = metric_config.get("Eval")
                    if metric_config is not None:
                        self.eval_metric_func = build_metrics(metric_config)
            elif self.eval_mode == "retrieval":
                if metric_config is None:
                    metric_config = [{"name": "Recallk", "topk": (1, 5)}]
                else:
                    metric_config = metric_config["Eval"]
                self.eval_metric_func = build_metrics(metric_config)
        else:
            self.eval_metric_func = None

        # build model
        self.model = build_model(self.config["Arch"])
        # set @to_static for benchmark, skip this by default.
        apply_to_static(self.config, self.model)

        # for slim
        self.pruner = get_pruner(self.config, self.model)
        self.quanter = get_quaner(self.config, self.model)

        # load_pretrain
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    self.model, self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    self.model, self.config["Global"]["pretrained_model"])

        # build optimizer
        if self.mode == 'train':
            self.optimizer, self.lr_sch = build_optimizer(
                self.config["Optimizer"], self.config["Global"]["epochs"],
                len(self.train_dataloader), [self.model])

        # for distributed
        self.config["Global"][
            "distributed"] = paddle.distributed.get_world_size() != 1
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
        if self.config["Global"]["distributed"]:
            self.model = paddle.DataParallel(self.model)

        # build postprocess for infer
        if self.mode == 'infer':
            self.preprocess_func = create_operators(self.config["Infer"][
                "transforms"])
            self.postprocess_func = build_postprocess(self.config["Infer"][
                "PostProcess"])

    def train(self):
        assert self.mode == "train"
        print_batch_step = self.config['Global']['print_batch_step']
        save_interval = self.config["Global"]["save_interval"]
        best_metric = {
            "metric": 0.0,
            "epoch": 0,
        }
        # key:
        # val: metrics list word
        self.output_info = dict()
        self.time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        # global iter counter
        self.global_step = 0

        if self.config["Global"]["checkpoints"] is not None:
            metric_info = init_model(self.config["Global"], self.model,
                                     self.optimizer)
            if metric_info is not None:
                best_metric.update(metric_info)

        # for amp training
        if self.amp:
            self.scaler = paddle.amp.GradScaler(
                init_loss_scaling=self.scale_loss,
                use_dynamic_loss_scaling=self.use_dynamic_loss_scaling)

        self.max_iter = len(self.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.train_dataloader)
        for epoch_id in range(best_metric["epoch"] + 1,
                              self.config["Global"]["epochs"] + 1):
            acc = 0.0
            # for one epoch train
            self.train_epoch_func(self, epoch_id, print_batch_step)

            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, self.output_info[key].avg)
                for key in self.output_info
            ])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.config["Global"]["epochs"], metric_msg))
            self.output_info.clear()

            # eval model and save model if possible
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_interval"] == 0:
                acc = self.eval(epoch_id)
                if acc > best_metric["metric"]:
                    best_metric["metric"] = acc
                    best_metric["epoch"] = epoch_id
                    save_load.save_model(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        model_name=self.config["Arch"]["name"],
                        prefix="best_model")
                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=acc,
                    step=epoch_id,
                    writer=self.vdl_writer)

                self.model.train()

            # save model
            if epoch_id % save_interval == 0:
                save_load.save_model(
                    self.model,
                    self.optimizer, {"metric": acc,
                                     "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config["Arch"]["name"],
                    prefix="epoch_{}".format(epoch_id))
                # save the latest model
                save_load.save_model(
                    self.model,
                    self.optimizer, {"metric": acc,
                                     "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config["Arch"]["name"],
                    prefix="latest")

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_func(self, epoch_id)
        self.model.train()
        return eval_result

    @paddle.no_grad()
    def infer(self):
        assert self.mode == "infer" and self.eval_mode == "classification"
        total_trainer = paddle.distributed.get_world_size()
        local_rank = paddle.distributed.get_rank()
        image_list = get_image_list(self.config["Infer"]["infer_imgs"])
        # data split
        image_list = image_list[local_rank::total_trainer]

        batch_size = self.config["Infer"]["batch_size"]
        self.model.eval()
        batch_data = []
        image_file_list = []
        for idx, image_file in enumerate(image_list):
            with open(image_file, 'rb') as f:
                x = f.read()
            for process in self.preprocess_func:
                x = process(x)
            batch_data.append(x)
            image_file_list.append(image_file)
            if len(batch_data) >= batch_size or idx == len(image_list) - 1:
                batch_tensor = paddle.to_tensor(batch_data)
                out = self.model(batch_tensor)
                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, dict):
                    out = out["output"]
                result = self.postprocess_func(out, image_file_list)
                print(result)
                batch_data.clear()
                image_file_list.clear()

    def export(self):
        assert self.mode == "export"
        use_multilabel = self.config["Global"].get("use_multilabel", False)
        model = ExportModel(self.config["Arch"], self.model, use_multilabel)
        if self.config["Global"]["pretrained_model"] is not None:
            load_dygraph_pretrain(model.base_model,
                                  self.config["Global"]["pretrained_model"])

        model.eval()
        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")
        if self.quanter:
            self.quanter.save_quantized_model(
                model.base_model,
                save_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config["Global"]["image_shape"],
                        dtype='float32')
                ])
        else:
            model = paddle.jit.to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config["Global"]["image_shape"],
                        dtype='float32')
                ])
            paddle.jit.save(model, save_path)


class ExportModel(nn.Layer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, model, use_multilabel):
        super().__init__()
        self.base_model = model
        # we should choose a final model to export
        if isinstance(self.base_model, DistillationModel):
            self.infer_model_name = config["infer_model_name"]
        else:
            self.infer_model_name = None

        self.infer_output_key = config.get("infer_output_key", None)
        if self.infer_output_key == "features" and isinstance(self.base_model,
                                                              RecModel):
            self.base_model.head = IdentityHead()
        if use_multilabel:
            self.out_act = nn.Sigmoid()
        else:
            if config.get("infer_add_softmax", True):
                self.out_act = nn.Softmax(axis=-1)
            else:
                self.out_act = None

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.base_model(x)
        if isinstance(x, list):
            x = x[0]
        if self.infer_model_name is not None:
            x = x[self.infer_model_name]
        if self.infer_output_key is not None:
            x = x[self.infer_output_key]
        if self.out_act is not None:
            x = self.out_act(x)
        return x
