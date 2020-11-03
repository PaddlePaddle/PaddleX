# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import yaml
import time
import pickle
import os
import os.path as osp
from functools import reduce
import paddle.fluid as fluid
from multiprocessing import Process, Queue
from .prune_config import get_prune_params
import paddlex.utils.logging as logging
from paddlex.utils import seconds_to_hms


def sensitivity(program,
                place,
                param_names,
                eval_func,
                sensitivities_file=None,
                pruned_ratios=None,
                scope=None):
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    if scope is None:
        scope = fluid.global_scope()
    else:
        scope = scope
    graph = GraphWrapper(program)
    sensitivities = load_sensitivities(sensitivities_file)

    if pruned_ratios is None:
        pruned_ratios = np.arange(0.1, 1, step=0.1)

    total_evaluate_iters = 0
    for name in param_names:
        if name not in sensitivities:
            sensitivities[name] = {}
            total_evaluate_iters += len(list(pruned_ratios))
        else:
            total_evaluate_iters += (
                len(list(pruned_ratios)) - len(sensitivities[name]))
    eta = '-'
    start_time = time.time()
    baseline = eval_func(graph.program)
    cost = time.time() - start_time
    eta = cost * (total_evaluate_iters - 1)
    current_iter = 1
    for name in sensitivities:
        for ratio in pruned_ratios:
            if ratio in sensitivities[name]:
                logging.debug('{}, {} has computed.'.format(name, ratio))
                continue

            progress = float(current_iter) / total_evaluate_iters
            progress = "%.2f%%" % (progress * 100)
            logging.info(
                "Total evaluate iters={}, current={}, progress={}, eta={}".
                format(total_evaluate_iters, current_iter, progress,
                       seconds_to_hms(
                           int(cost * (total_evaluate_iters - current_iter)))),
                use_color=True)
            current_iter += 1

            pruner = Pruner()
            logging.info("sensitive - param: {}; ratios: {}".format(name,
                                                                    ratio))
            pruned_program, param_backup, _ = pruner.prune(
                program=graph.program,
                scope=scope,
                params=[name],
                ratios=[ratio],
                place=place,
                lazy=True,
                only_graph=False,
                param_backup=True)
            pruned_metric = eval_func(pruned_program)
            loss = (baseline - pruned_metric) / baseline
            logging.info("pruned param: {}; {}; loss={}".format(name, ratio,
                                                                loss))

            sensitivities[name][ratio] = loss

            with open(sensitivities_file, 'wb') as f:
                pickle.dump(sensitivities, f)

            for param_name in param_backup.keys():
                param_t = scope.find_var(param_name).get_tensor()
                param_t.set(param_backup[param_name], place)
    return sensitivities


def channel_prune(program,
                  prune_names,
                  prune_ratios,
                  place,
                  only_graph=False,
                  scope=None):
    """通道裁剪。

    Args:
        program (paddle.fluid.Program): 需要裁剪的Program，Program的具体介绍可参见
            https://paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/program.html#program。
        prune_names (list): 由裁剪参数名组成的参数列表。
        prune_ratios (list): 由裁剪率组成的参数列表，与prune_names中的参数列表意义对应。
        place (paddle.fluid.CUDAPlace/paddle.fluid.CPUPlace): 运行设备。
        only_graph (bool): 是否只修改网络图，当为False时代表同时修改网络图和
            scope（全局作用域）中的参数。默认为False。

    Returns:
        paddle.fluid.Program: 裁剪后的Program。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    prog_var_shape_dict = {}
    for var in program.list_vars():
        try:
            prog_var_shape_dict[var.name] = var.shape
        except Exception:
            pass
    index = 0
    for param, ratio in zip(prune_names, prune_ratios):
        origin_num = prog_var_shape_dict[param][0]
        pruned_num = int(round(origin_num * ratio))
        while origin_num == pruned_num:
            ratio -= 0.1
            pruned_num = int(round(origin_num * (ratio)))
            prune_ratios[index] = ratio
        index += 1
    if scope is None:
        scope = fluid.global_scope()
    pruner = Pruner()
    program, _, _ = pruner.prune(
        program,
        scope,
        params=prune_names,
        ratios=prune_ratios,
        place=place,
        lazy=False,
        only_graph=only_graph,
        param_backup=False,
        param_shape_backup=False)
    return program


def prune_program(model, prune_params_ratios=None):
    """根据裁剪参数和裁剪率裁剪Program。

    1. 裁剪训练Program和测试Program。
    2. 使用裁剪后的Program更新模型中的train_prog和test_prog。
    【注意】Program的具体介绍可参见
            https://paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/program.html#program。

    Args:
        model (paddlex.cv.models): paddlex中的模型。
        prune_params_ratios (dict): 由裁剪参数名和裁剪率组成的字典，当为None时
            使用默认裁剪参数名和裁剪率。默认为None。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    assert model.status == 'Normal', 'Only the models saved while training are supported!'
    place = model.places[0]
    train_prog = model.train_prog
    eval_prog = model.test_prog
    valid_prune_names = get_prune_params(model)
    assert set(list(prune_params_ratios.keys())) & set(valid_prune_names), \
        "All params in 'prune_params_ratios' can't be pruned!"
    prune_names = list(
        set(list(prune_params_ratios.keys())) & set(valid_prune_names))
    prune_ratios = [
        prune_params_ratios[prune_name] for prune_name in prune_names
    ]
    model.train_prog = channel_prune(
        train_prog, prune_names, prune_ratios, place, scope=model.scope)
    model.test_prog = channel_prune(
        eval_prog,
        prune_names,
        prune_ratios,
        place,
        only_graph=True,
        scope=model.scope)


def update_program(program, model_dir, place, scope=None):
    """根据裁剪信息更新Program和参数。

    Args:
        program (paddle.fluid.Program): 需要更新的Program，Program的具体介绍可参见
            https://paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/program.html#program。
        model_dir (str): 模型存储路径。
        place (paddle.fluid.CUDAPlace/paddle.fluid.CPUPlace): 运行设备。

    Returns:
        paddle.fluid.Program: 更新后的Program。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    graph = GraphWrapper(program)
    with open(osp.join(model_dir, "prune.yml")) as f:
        shapes = yaml.load(f.read(), Loader=yaml.Loader)
    for param, shape in shapes.items():
        graph.var(param).set_shape(shape)
    if scope is None:
        scope = fluid.global_scope()
    for block in program.blocks:
        for param in block.all_parameters():
            if param.name in shapes:
                param_tensor = scope.find_var(param.name).get_tensor()
                param_tensor.set(
                    np.zeros(list(shapes[param.name])).astype('float32'), place)
    graph.update_groups_of_conv()
    graph.infer_shape()
    return program


def cal_params_sensitivities(model, save_file, eval_dataset, batch_size=8):
    """计算模型中可裁剪卷积Kernel的敏感度。

       1. 获取模型中可裁剪卷积Kernel的名称。
       2. 计算每个可裁剪卷积Kernel不同裁剪率下的敏感度。
       【注意】卷积的敏感度是指在不同裁剪率下评估数据集预测精度的损失，
           通过得到的敏感度，可以决定最终模型需要裁剪的参数列表和各裁剪参数对应的裁剪率。

    Args:
        model (paddlex.cv.models): paddlex中的模型。
        save_file (str): 计算的得到的sensetives文件存储路径。
        eval_dataset (paddlex.datasets): 验证数据读取器。
        batch_size (int): 验证数据批大小。默认为8。

    Returns:
        dict: 由参数名和不同裁剪率下敏感度组成的字典。存储的信息如下：
        .. code-block:: python

            {"weight_0":
                {0.1: 0.22,
                 0.2: 0.33
                },
             "weight_1":
                {0.1: 0.21,
                 0.2: 0.4
                }
            }

            其中``weight_0``是卷积Kernel名；``sensitivities['weight_0']``是一个字典，key是裁剪率，value是敏感度。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    assert model.status == 'Normal', 'Only the models saved while training are supported!'
    if os.path.exists(save_file):
        os.remove(save_file)

    prune_names = get_prune_params(model)

    def eval_for_prune(program):
        eval_metrics = model.evaluate(
            eval_dataset=eval_dataset,
            batch_size=batch_size,
            return_details=False)
        primary_key = list(eval_metrics.keys())[0]
        return eval_metrics[primary_key]

    sensitivitives = sensitivity(
        model.test_prog,
        model.places[0],
        prune_names,
        eval_for_prune,
        sensitivities_file=save_file,
        pruned_ratios=list(np.arange(0.1, 1, 0.1)),
        scope=model.scope)
    return sensitivitives


def analysis(model, dataset, batch_size=8, save_file='./model.sensi.data'):
    return cal_params_sensitivities(
        model, eval_dataset=dataset, batch_size=batch_size, save_file=save_file)


def get_params_ratios(sensitivities_file, eval_metric_loss=0.05):
    """根据设定的精度损失容忍度metric_loss_thresh和计算保存的模型参数敏感度信息文件sensetive_file，
        获取裁剪的参数配置。

        【注意】metric_loss_thresh并不确保最终裁剪后的模型在fine-tune后的模型效果，仅为预估值。

    Args:
        sensitivities_file (str): 敏感度文件存储路径。
        eval_metric_loss (float): 可容忍的精度损失。默认为0.05。

    Returns:
        dict: 由参数名和裁剪率组成的字典。存储的信息如下：
        .. code-block:: python

            {"weight_0": 0.1,
             "weight_1": 0.2
            }

            其中key是卷积Kernel名；value是裁剪率。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    if not osp.exists(sensitivities_file):
        raise Exception('The sensitivities file is not exists!')
    sensitivitives = paddleslim.prune.load_sensitivities(sensitivities_file)
    params_ratios = paddleslim.prune.get_ratios_by_loss(sensitivitives,
                                                        eval_metric_loss)
    return params_ratios


def cal_model_size(program,
                   place,
                   sensitivities_file,
                   eval_metric_loss=0.05,
                   scope=None):
    """在可容忍的精度损失下，计算裁剪后模型大小相对于当前模型大小的比例。

    Args:
        program (paddle.fluid.Program): 需要裁剪的Program，Program的具体介绍可参见
            https://paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/basic_concept/program.html#program。
        place (paddle.fluid.CUDAPlace/paddle.fluid.CPUPlace): 运行设备。
        sensitivities_file (str): 敏感度文件存储路径。
        eval_metric_loss (float): 可容忍的精度损失。默认为0.05。

    Returns:
        float: 裁剪后模型大小相对于当前模型大小的比例。
    """
    import paddleslim
    from paddleslim.prune import Pruner, load_sensitivities
    from paddleslim.core import GraphWrapper

    prune_params_ratios = get_params_ratios(sensitivities_file,
                                            eval_metric_loss)
    prog_var_shape_dict = {}
    for var in program.list_vars():
        try:
            prog_var_shape_dict[var.name] = var.shape
        except Exception:
            pass
    for param, ratio in prune_params_ratios.items():
        origin_num = prog_var_shape_dict[param][0]
        pruned_num = int(round(origin_num * ratio))
        while origin_num == pruned_num:
            ratio -= 0.1
            pruned_num = int(round(origin_num * (ratio)))
            prune_params_ratios[param] = ratio
    prune_program = channel_prune(
        program,
        list(prune_params_ratios.keys()),
        list(prune_params_ratios.values()),
        place,
        only_graph=True,
        scope=scope)
    origin_size = 0
    new_size = 0
    for var in program.list_vars():
        name = var.name
        shape = var.shape
        for prune_block in prune_program.blocks:
            if prune_block.has_var(name):
                prune_var = prune_block.var(name)
                prune_shape = prune_var.shape
                break
        origin_size += reduce(lambda x, y: x * y, shape)
        new_size += reduce(lambda x, y: x * y, prune_shape)
    return (new_size * 1.0) / origin_size
