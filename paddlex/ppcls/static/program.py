# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np

from collections import OrderedDict

import paddle
import paddle.nn.functional as F

from paddle.distributed import fleet
from paddle.distributed.fleet import DistributedStrategy

# from paddlex.ppcls.optimizer import OptimizerBuilder
# from paddlex.ppcls.optimizer.learning_rate import LearningRateBuilder

from paddlex.ppcls.arch import build_model
from paddlex.ppcls.loss import build_loss
from paddlex.ppcls.metric import build_metrics
from paddlex.ppcls.optimizer import build_optimizer
from paddlex.ppcls.optimizer import build_lr_scheduler

from paddlex.ppcls.utils.misc import AverageMeter
from paddlex.ppcls.utils import logger, profiler


def create_feeds(image_shape, use_mix=False, class_num=None, dtype="float32"):
    """
    Create feeds as model input

    Args:
        image_shape(list[int]): model input shape, such as [3, 224, 224]
        use_mix(bool): whether to use mix(include mixup, cutmix, fmix)
        class_num(int): the class number of network, required if use_mix

    Returns:
        feeds(dict): dict of model input variables
    """
    feeds = OrderedDict()
    feeds['data'] = paddle.static.data(
        name="data", shape=[None] + image_shape, dtype=dtype)

    if use_mix:
        if class_num is None:
            msg = "When use MixUp, CutMix and so on, you must set class_num."
            logger.error(msg)
            raise Exception(msg)
        feeds['target'] = paddle.static.data(
            name="target", shape=[None, class_num], dtype="float32")
    else:
        feeds['label'] = paddle.static.data(
            name="label", shape=[None, 1], dtype="int64")

    return feeds


def create_fetchs(out,
                  feeds,
                  architecture,
                  topk=5,
                  epsilon=None,
                  class_num=None,
                  use_mix=False,
                  config=None,
                  mode="Train"):
    """
    Create fetchs as model outputs(included loss and measures),
    will call create_loss and create_metric(if use_mix).
    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables.
            If use mix_up, it will not include label.
        architecture(dict): architecture information,
            name(such as ResNet50) is needed
        topk(int): usually top5
        epsilon(float): parameter for label smoothing, 0.0 <= epsilon <= 1.0
        class_num(int): the class number of network, required if use_mix
        use_mix(bool): whether to use mix(include mixup, cutmix, fmix)
        config(dict): model config

    Returns:
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    fetchs = OrderedDict()
    # build loss
    if use_mix:
        if class_num is None:
            msg = "When use MixUp, CutMix and so on, you must set class_num."
            logger.error(msg)
            raise Exception(msg)
        target = paddle.reshape(feeds['target'], [-1, class_num])
    else:
        target = paddle.reshape(feeds['label'], [-1, 1])

    loss_func = build_loss(config["Loss"][mode])
    loss_dict = loss_func(out, target)

    loss_out = loss_dict["loss"]
    fetchs['loss'] = (loss_out, AverageMeter('loss', '7.4f', need_avg=True))

    # build metric
    if not use_mix:
        metric_func = build_metrics(config["Metric"][mode])

        metric_dict = metric_func(out, target)

        for key in metric_dict:
            if mode != "Train" and paddle.distributed.get_world_size() > 1:
                paddle.distributed.all_reduce(
                    metric_dict[key], op=paddle.distributed.ReduceOp.SUM)
                metric_dict[key] = metric_dict[
                    key] / paddle.distributed.get_world_size()

            fetchs[key] = (metric_dict[key], AverageMeter(
                key, '7.4f', need_avg=True))

    return fetchs


def create_optimizer(config, step_each_epoch):
    # create learning_rate instance
    optimizer, lr_sch = build_optimizer(
        config["Optimizer"], config["Global"]["epochs"], step_each_epoch)
    return optimizer, lr_sch


def create_strategy(config):
    """
    Create build strategy and exec strategy.

    Args:
        config(dict): config

    Returns:
        build_strategy: build strategy
        exec_strategy: exec strategy
    """
    build_strategy = paddle.static.BuildStrategy()
    exec_strategy = paddle.static.ExecutionStrategy()

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = (
        10000
        if 'AMP' in config and config.AMP.get("use_pure_fp16", False) else 10)

    fuse_op = True if 'AMP' in config else False

    fuse_bn_act_ops = config.get('fuse_bn_act_ops', fuse_op)
    fuse_elewise_add_act_ops = config.get('fuse_elewise_add_act_ops', fuse_op)
    fuse_bn_add_act_ops = config.get('fuse_bn_add_act_ops', fuse_op)
    enable_addto = config.get('enable_addto', fuse_op)

    build_strategy.fuse_bn_act_ops = fuse_bn_act_ops
    build_strategy.fuse_elewise_add_act_ops = fuse_elewise_add_act_ops
    build_strategy.fuse_bn_add_act_ops = fuse_bn_add_act_ops
    build_strategy.enable_addto = enable_addto

    return build_strategy, exec_strategy


def dist_optimizer(config, optimizer):
    """
    Create a distributed optimizer based on a normal optimizer

    Args:
        config(dict):
        optimizer(): a normal optimizer

    Returns:
        optimizer: a distributed optimizer
    """
    build_strategy, exec_strategy = create_strategy(config)

    dist_strategy = DistributedStrategy()
    dist_strategy.execution_strategy = exec_strategy
    dist_strategy.build_strategy = build_strategy

    dist_strategy.nccl_comm_num = 1
    dist_strategy.fuse_all_reduce_ops = True
    dist_strategy.fuse_grad_size_in_MB = 16
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

    return optimizer


def mixed_precision_optimizer(config, optimizer):
    if 'AMP' in config:
        amp_cfg = config.AMP if config.AMP else dict()
        scale_loss = amp_cfg.get('scale_loss', 1.0)
        use_dynamic_loss_scaling = amp_cfg.get('use_dynamic_loss_scaling',
                                               False)
        use_pure_fp16 = amp_cfg.get('use_pure_fp16', False)
        optimizer = paddle.static.amp.decorate(
            optimizer,
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
            use_pure_fp16=use_pure_fp16,
            use_fp16_guard=True)

    return optimizer


def build(config,
          main_prog,
          startup_prog,
          class_num=None,
          step_each_epoch=100,
          is_train=True,
          is_distributed=True):
    """
    Build a program using a model and an optimizer
        1. create feeds
        2. create a dataloader
        3. create a model
        4. create fetchs
        5. create an optimizer

    Args:
        config(dict): config
        main_prog(): main program
        startup_prog(): startup program
        class_num(int): the class number of network, required if use_mix
        is_train(bool): train or eval
        is_distributed(bool): whether to use distributed training method

    Returns:
        dataloader(): a bridge between the model and the data
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    with paddle.static.program_guard(main_prog, startup_prog):
        with paddle.utils.unique_name.guard():
            mode = "Train" if is_train else "Eval"
            use_mix = "batch_transform_ops" in config["DataLoader"][mode][
                "dataset"]
            feeds = create_feeds(
                config["Global"]["image_shape"],
                use_mix,
                class_num=class_num,
                dtype="float32")

            # build model
            # data_format should be assigned in arch-dict
            input_image_channel = config["Global"]["image_shape"][
                0]  # default as [3, 224, 224]
            model = build_model(config["Arch"])
            out = model(feeds["data"])
            # end of build model

            fetchs = create_fetchs(
                out,
                feeds,
                config["Arch"],
                epsilon=config.get('ls_epsilon'),
                class_num=class_num,
                use_mix=use_mix,
                config=config,
                mode=mode)
            lr_scheduler = None
            optimizer = None
            if is_train:
                optimizer, lr_scheduler = build_optimizer(
                    config["Optimizer"], config["Global"]["epochs"],
                    step_each_epoch)
                optimizer = mixed_precision_optimizer(config, optimizer)
                if is_distributed:
                    optimizer = dist_optimizer(config, optimizer)
                optimizer.minimize(fetchs['loss'][0])
    return fetchs, lr_scheduler, feeds, optimizer


def compile(config, program, loss_name=None, share_prog=None):
    """
    Compile the program

    Args:
        config(dict): config
        program(): the program which is wrapped by
        loss_name(str): loss name
        share_prog(): the shared program, used for evaluation during training

    Returns:
        compiled_program(): a compiled program
    """
    build_strategy, exec_strategy = create_strategy(config)

    compiled_program = paddle.static.CompiledProgram(
        program).with_data_parallel(
            share_vars_from=share_prog,
            loss_name=loss_name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    return compiled_program


total_step = 0


def run(dataloader,
        exe,
        program,
        feeds,
        fetchs,
        epoch=0,
        mode='train',
        config=None,
        vdl_writer=None,
        lr_scheduler=None,
        profiler_options=None):
    """
    Feed data to the model and fetch the measures and loss

    Args:
        dataloader(paddle io dataloader):
        exe():
        program():
        fetchs(dict): dict of measures and the loss
        epoch(int): epoch of training or evaluation
        model(str): log only

    Returns:
    """
    fetch_list = [f[0] for f in fetchs.values()]
    metric_dict = OrderedDict([("lr", AverageMeter(
        'lr', 'f', postfix=",", need_avg=False))])

    for k in fetchs:
        metric_dict[k] = fetchs[k][1]

    metric_dict["batch_time"] = AverageMeter(
        'batch_cost', '.5f', postfix=" s,")
    metric_dict["reader_time"] = AverageMeter(
        'reader_cost', '.5f', postfix=" s,")

    for m in metric_dict.values():
        m.reset()

    use_dali = config["Global"].get('use_dali', False)
    tic = time.time()

    if not use_dali:
        dataloader = dataloader()

    idx = 0
    batch_size = None
    while True:
        # The DALI maybe raise RuntimeError for some particular images, such as ImageNet1k/n04418357_26036.JPEG
        try:
            batch = next(dataloader)
        except StopIteration:
            break
        except RuntimeError:
            logger.warning(
                "Except RuntimeError when reading data from dataloader, try to read once again..."
            )
            continue
        idx += 1
        # ignore the warmup iters
        if idx == 5:
            metric_dict["batch_time"].reset()
            metric_dict["reader_time"].reset()

        metric_dict['reader_time'].update(time.time() - tic)

        profiler.add_profiler_step(profiler_options)

        if use_dali:
            batch_size = batch[0]["data"].shape()[0]
            feed_dict = batch[0]
        else:
            batch_size = batch[0].shape()[0]
            feed_dict = {
                key.name: batch[idx]
                for idx, key in enumerate(feeds.values())
            }

        metrics = exe.run(program=program,
                          feed=feed_dict,
                          fetch_list=fetch_list)

        for name, m in zip(fetchs.keys(), metrics):
            metric_dict[name].update(np.mean(m), batch_size)
        metric_dict["batch_time"].update(time.time() - tic)
        if mode == "train":
            metric_dict['lr'].update(lr_scheduler.get_lr())

        fetchs_str = ' '.join([
            str(metric_dict[key].mean)
            if "time" in key else str(metric_dict[key].value)
            for key in metric_dict
        ])
        ips_info = " ips: {:.5f} images/sec.".format(
            batch_size / metric_dict["batch_time"].avg)
        fetchs_str += ips_info

        if lr_scheduler is not None:
            lr_scheduler.step()

        if vdl_writer:
            global total_step
            logger.scaler('loss', metrics[0][0], total_step, vdl_writer)
            total_step += 1
        if mode == 'eval':
            if idx % config.get('print_interval', 10) == 0:
                logger.info("{:s} step:{:<4d} {:s}".format(mode, idx,
                                                           fetchs_str))
        else:
            epoch_str = "epoch:{:<3d}".format(epoch)
            step_str = "{:s} step:{:<4d}".format(mode, idx)

            if idx % config.get('print_interval', 10) == 0:
                logger.info("{:s} {:s} {:s}".format(epoch_str, step_str,
                                                    fetchs_str))

        tic = time.time()

    end_str = ' '.join([str(m.mean) for m in metric_dict.values()] +
                       [metric_dict["batch_time"].total])
    ips_info = "ips: {:.5f} images/sec.".format(batch_size /
                                                metric_dict["batch_time"].avg)
    if mode == 'eval':
        logger.info("END {:s} {:s} {:s}".format(mode, end_str, ips_info))
    else:
        end_epoch_str = "END epoch:{:<3d}".format(epoch)
        logger.info("{:s} {:s} {:s} {:s}".format(end_epoch_str, mode, end_str,
                                                 ips_info))
    if use_dali:
        dataloader.reset()

    # return top1_acc in order to save the best model
    if mode == 'eval':
        return fetchs["top1"][1].avg
