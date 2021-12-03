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

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import paddle
from paddle.distributed import fleet
from visualdl import LogWriter

from paddlex.ppcls.data import build_dataloader
from paddlex.ppcls.utils.config import get_config, print_config
from paddlex.ppcls.utils import logger
from paddlex.ppcls.utils.logger import init_logger
from paddlex.ppcls.static.save_load import init_model, save_model
from paddlex.ppcls.static import program


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet/ResNet50.yaml',
        help='config file path')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args


def main(args):
    """
    all the config of training paradigm should be in config["Global"]
    """
    config = get_config(args.config, overrides=args.override, show=False)
    global_config = config["Global"]

    mode = "train"

    log_file = os.path.join(global_config['output_dir'],
                            config["Arch"]["name"], f"{mode}.log")
    init_logger(name='root', log_file=log_file)
    print_config(config)

    if global_config.get("is_distributed", True):
        fleet.init(is_collective=True)
    # assign the device
    use_gpu = global_config.get("use_gpu", True)
    # amp related config
    if 'AMP' in config:
        AMP_RELATED_FLAGS_SETTING = {
            'FLAGS_cudnn_exhaustive_search': "1",
            'FLAGS_conv_workspace_size_limit': "1500",
            'FLAGS_cudnn_batchnorm_spatial_persistent': "1",
            'FLAGS_max_indevice_grad_add': "8",
            "FLAGS_cudnn_batchnorm_spatial_persistent": "1",
        }
        for k in AMP_RELATED_FLAGS_SETTING:
            os.environ[k] = AMP_RELATED_FLAGS_SETTING[k]

    use_xpu = global_config.get("use_xpu", False)
    use_npu = global_config.get("use_npu", False)
    assert (
        use_gpu and use_xpu and use_npu
    ) is not True, "gpu, xpu and npu can not be true in the same time in static mode!"

    if use_gpu:
        device = paddle.set_device('gpu')
    elif use_xpu:
        device = paddle.set_device('xpu')
    elif use_npu:
        device = paddle.set_device('npu')
    else:
        device = paddle.set_device('cpu')

    # visualDL
    vdl_writer = None
    if global_config["use_visualdl"]:
        vdl_dir = os.path.join(global_config["output_dir"], "vdl")
        vdl_writer = LogWriter(vdl_dir)

    # build dataloader
    eval_dataloader = None
    use_dali = global_config.get('use_dali', False)

    class_num = config["Arch"].get("class_num", None)
    config["DataLoader"].update({"class_num": class_num})
    train_dataloader = build_dataloader(
        config["DataLoader"], "Train", device=device, use_dali=use_dali)
    if global_config["eval_during_train"]:
        eval_dataloader = build_dataloader(
            config["DataLoader"], "Eval", device=device, use_dali=use_dali)

    step_each_epoch = len(train_dataloader)

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = paddle.static.Program()
    train_prog = paddle.static.Program()

    best_top1_acc = 0.0  # best top1 acc record

    train_fetchs, lr_scheduler, train_feeds, optimizer = program.build(
        config,
        train_prog,
        startup_prog,
        class_num,
        step_each_epoch=step_each_epoch,
        is_train=True,
        is_distributed=global_config.get("is_distributed", True))

    if global_config["eval_during_train"]:
        eval_prog = paddle.static.Program()
        eval_fetchs, _, eval_feeds, _ = program.build(
            config,
            eval_prog,
            startup_prog,
            is_train=False,
            is_distributed=global_config.get("is_distributed", True))
        # clone to prune some content which is irrelevant in eval_prog
        eval_prog = eval_prog.clone(for_test=True)

    # create the "Executor" with the statement of which device
    exe = paddle.static.Executor(device)
    # Parameter initialization
    exe.run(startup_prog)
    # load pretrained models or checkpoints
    init_model(global_config, train_prog, exe)

    if 'AMP' in config and config.AMP.get("use_pure_fp16", False):
        optimizer.amp_init(
            device,
            scope=paddle.static.global_scope(),
            test_program=eval_prog
            if global_config["eval_during_train"] else None)

    if not global_config.get("is_distributed", True):
        compiled_train_prog = program.compile(
            config, train_prog, loss_name=train_fetchs["loss"][0].name)
    else:
        compiled_train_prog = train_prog

    if eval_dataloader is not None:
        compiled_eval_prog = program.compile(config, eval_prog)

    for epoch_id in range(global_config["epochs"]):
        # 1. train with train dataset
        program.run(train_dataloader, exe, compiled_train_prog, train_feeds,
                    train_fetchs, epoch_id, 'train', config, vdl_writer,
                    lr_scheduler, args.profiler_options)
        # 2. evaate with eval dataset
        if global_config["eval_during_train"] and epoch_id % global_config[
                "eval_interval"] == 0:
            top1_acc = program.run(eval_dataloader, exe, compiled_eval_prog,
                                   eval_feeds, eval_fetchs, epoch_id, "eval",
                                   config)
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                    best_top1_acc, epoch_id)
                logger.info(message)
                if epoch_id % global_config["save_interval"] == 0:

                    model_path = os.path.join(global_config["output_dir"],
                                              config["Arch"]["name"])
                    save_model(train_prog, model_path, "best_model")

        # 3. save the persistable model
        if epoch_id % global_config["save_interval"] == 0:
            model_path = os.path.join(global_config["output_dir"],
                                      config["Arch"]["name"])
            save_model(train_prog, model_path, epoch_id)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    main(args)
