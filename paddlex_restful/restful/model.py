# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import time
import os
import shutil
import pickle
from os import path as osp
from .utils import set_folder_status, TaskStatus, copy_pretrained_model, PretrainedModelStatus
from . import workspace_pb2 as w


def list_pretrained_models(workspace):
    """列出预训练模型列表
    """
    pretrained_model_list = list()
    for id in workspace.pretrained_models:
        pretrained_model = workspace.pretrained_models[id]
        model_id = pretrained_model.id
        model_name = pretrained_model.name
        model_model = pretrained_model.model
        model_type = pretrained_model.type
        model_pid = pretrained_model.pid
        model_tid = pretrained_model.tid
        model_create_time = pretrained_model.create_time
        model_path = pretrained_model.path
        attr = {
            'id': model_id,
            'name': model_name,
            'model': model_model,
            'type': model_type,
            'pid': model_pid,
            'tid': model_tid,
            'create_time': model_create_time,
            'path': model_path
        }
        pretrained_model_list.append(attr)

    return {'status': 1, "pretrained_models": pretrained_model_list}


def create_pretrained_model(data, workspace, monitored_processes):
    """根据request创建预训练模型。

    Args:
        data为dict,key包括
        'pid'所属项目id, 'tid'所属任务id,'name'预训练模型名称,
        'source_path' 原模型路径, 'eval_results'（可选） 评估结果数据
    """
    time_array = time.localtime(time.time())
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_pretrained_model_id + 1
    workspace.max_pretrained_model_id = id
    if id < 10000:
        id = 'PM%04d' % id
    else:
        id = 'PM{}'.format(id)
    pid = data['pid']
    tid = data['tid']
    name = data['name']
    source_path = data['source_path']
    assert pid in workspace.projects, "【预训练模型创建】项目ID'{}'不存在.".format(pid)
    assert tid in workspace.tasks, "【预训练模型创建】任务ID'{}'不存在.".format(tid)
    assert not id in workspace.pretrained_models, "【预训练模型创建】预训练模型'{}'已经被占用.".format(
        id)
    assert osp.exists(source_path), "原模型路径不存在: {}".format(source_path)
    path = osp.join(workspace.path, 'pretrain', id)
    if not osp.exists(path):
        os.makedirs(path)
    set_folder_status(path, PretrainedModelStatus.XPINIT)
    params = {'tid': tid}
    from .project.task import get_task_params
    ret = get_task_params(params, workspace)
    train_params = ret['train']
    model_structure = train_params.model
    if hasattr(train_params, "backbone"):
        model_structure = "{}-{}".format(model_structure,
                                         train_params.backbone)
    if hasattr(train_params, "with_fpn"):
        if train_params.with_fpn:
            model_structure = "{}-{}".format(model_structure, "WITH_FPN")

    pm = w.PretrainedModel(
        id=id,
        name=name,
        model=model_structure,
        type=workspace.projects[pid].type,
        pid=pid,
        tid=tid,
        create_time=create_time,
        path=path)
    workspace.pretrained_models[id].CopyFrom(pm)
    # 保存评估结果
    if 'eval_results' in data:
        with open(osp.join(source_path, "eval_res.pkl"), "wb") as f:
            pickle.dump(data['eval_results'], f)
    # 拷贝训练参数文件
    task_path = workspace.tasks[tid].path
    task_params_path = osp.join(task_path, 'params.pkl')
    if osp.exists(task_params_path):
        shutil.copy(task_params_path, path)
    # 拷贝数据集信息文件
    did = workspace.projects[pid].did
    dataset_path = workspace.datasets[did].path
    dataset_info_path = osp.join(dataset_path, "statis.pkl")
    if osp.exists(dataset_info_path):
        # 写入部分数据集信息
        with open(dataset_info_path, "rb") as f:
            dataset_info_dict = pickle.load(f)
        dataset_info_dict['name'] = workspace.datasets[did].name
        dataset_info_dict['desc'] = workspace.datasets[did].desc
        with open(dataset_info_path, "wb") as f:
            pickle.dump(dataset_info_dict, f)
        shutil.copy(dataset_info_path, path)

    # copy from source_path to path
    proc = copy_pretrained_model(source_path, path)
    monitored_processes.put(proc.pid)
    return {'status': 1, 'pmid': id}


def delete_pretrained_model(data, workspace):
    """删除pretrained_model。

    Args:
        data为dict,
        key包括'pmid'预训练模型id
    """
    pmid = data['pmid']
    assert pmid in workspace.pretrained_models, "预训练模型ID'{}'不存在.".format(pmid)
    if osp.exists(workspace.pretrained_models[pmid].path):
        shutil.rmtree(workspace.pretrained_models[pmid].path)
    del workspace.pretrained_models[pmid]
    return {'status': 1}


def create_exported_model(data, workspace):
    """根据request创建已发布模型。
    Args:
        data为dict,key包括
        'pid'所属项目id, 'tid'所属任务id,'name'已发布模型名称,
        'path' 模型路径, 'exported_type' 已发布模型类型,
    """
    time_array = time.localtime(time.time())
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    emid = workspace.max_exported_model_id + 1
    workspace.max_exported_model_id = emid
    if emid < 10000:
        emid = 'EM%04d' % emid
    else:
        emid = 'EM{}'.format(emid)
    pid = data['pid']
    tid = data['tid']
    name = data['name']
    path = data['path']
    exported_type = data['exported_type']
    assert pid in workspace.projects, "【已发布模型创建】项目ID'{}'不存在.".format(pid)
    assert tid in workspace.tasks, "【已发布模型创建】任务ID'{}'不存在.".format(tid)
    assert emid not in workspace.exported_models, "【已发布模型创建】已发布模型'{}'已经被占用.".format(
        emid)
    #assert osp.exists(path), "已发布模型路径不存在: {}".format(path)
    if not osp.exists(path):
        os.makedirs(path)
    task_path = workspace.tasks[tid].path
    # 拷贝评估结果
    eval_res_path = osp.join(task_path, 'eval_res.pkl')
    if osp.exists(eval_res_path):
        shutil.copy(eval_res_path, path)
    # 拷贝训练参数文件
    task_params_path = osp.join(task_path, 'params.pkl')
    if osp.exists(task_params_path):
        shutil.copy(task_params_path, path)
    # 拷贝数据集信息文件
    did = workspace.projects[pid].did
    dataset_path = workspace.datasets[did].path
    dataset_info_path = osp.join(dataset_path, "statis.pkl")
    if osp.exists(dataset_info_path):
        # 写入部分数据集信息
        with open(dataset_info_path, "rb") as f:
            dataset_info_dict = pickle.load(f)
        dataset_info_dict['name'] = workspace.datasets[did].name
        dataset_info_dict['desc'] = workspace.datasets[did].desc
        with open(dataset_info_path, "wb") as f:
            pickle.dump(dataset_info_dict, f)
        shutil.copy(dataset_info_path, path)
    from .project.task import get_task_params
    params = {'tid': tid}
    ret = get_task_params(params, workspace)
    train_params = ret['train']
    model_structure = train_params.model
    if hasattr(train_params, "backbone"):
        model_structure = "{}-{}".format(model_structure,
                                         train_params.backbone)
    if hasattr(train_params, "with_fpn"):
        if train_params.with_fpn:
            model_structure = "{}-{}".format(model_structure, "WITH_FPN")

    em = w.ExportedModel(
        id=emid,
        name=name,
        model=model_structure,
        type=workspace.projects[pid].type,
        pid=pid,
        tid=tid,
        create_time=create_time,
        path=path,
        exported_type=exported_type)

    workspace.exported_models[emid].CopyFrom(em)
    return {'status': 1, 'emid': emid}


def list_exported_models(workspace):
    """列出预训练模型列表，可根据request中的参数进行筛选

    Args:
    """
    exported_model_list = list()
    for id in workspace.exported_models:
        exported_model = workspace.exported_models[id]
        model_id = exported_model.id
        model_name = exported_model.name
        model_model = exported_model.model
        model_type = exported_model.type
        model_pid = exported_model.pid
        model_tid = exported_model.tid
        model_create_time = exported_model.create_time
        model_path = exported_model.path
        model_exported_type = exported_model.exported_type
        attr = {
            'id': model_id,
            'name': model_name,
            'model': model_model,
            'type': model_type,
            'pid': model_pid,
            'tid': model_tid,
            'create_time': model_create_time,
            'path': model_path,
            'exported_type': model_exported_type
        }
        if model_tid in workspace.tasks:
            from .project.task import get_export_status
            params = {'tid': model_tid}
            results = get_export_status(params, workspace)
            if results['export_status'] == TaskStatus.XEXPORTED:
                exported_model_list.append(attr)
        else:
            exported_model_list.append(attr)
    return {'status': 1, "exported_models": exported_model_list}


def delete_exported_model(data, workspace):
    """删除exported_model。

    Args:
        data为dict,
        key包括'emid'已发布模型id
    """
    emid = data['emid']
    assert emid in workspace.exported_models, "已发布模型模型ID'{}'不存在.".format(emid)
    if osp.exists(workspace.exported_models[emid].path):
        shutil.rmtree(workspace.exported_models[emid].path)
    del workspace.exported_models[emid]
    return {'status': 1}


def get_model_details(data, workspace):
    """获取模型详情。

    Args:
        data为dict,
        key包括'mid'模型id
    """
    mid = data['mid']
    if mid in workspace.pretrained_models:
        model_path = workspace.pretrained_models[mid].path
    elif mid in workspace.exported_models:
        model_path = workspace.exported_models[mid].path
    else:
        raise "模型{}不存在".format(mid)
    dataset_file = osp.join(model_path, 'statis.pkl')
    dataset_info = pickle.load(open(dataset_file, 'rb'))
    dataset_attr = {
        'name': dataset_info['name'],
        'desc': dataset_info['desc'],
        'labels': dataset_info['labels'],
        'train_num': len(dataset_info['train_files']),
        'val_num': len(dataset_info['val_files']),
        'test_num': len(dataset_info['test_files'])
    }
    task_params_file = osp.join(model_path, 'params.pkl')
    task_params = pickle.load(open(task_params_file, 'rb'))
    eval_result_file = osp.join(model_path, 'eval_res.pkl')
    eval_result = pickle.load(open(eval_result_file, 'rb'))
    #model_file = {'task_attr': task_params_file, 'eval_result': eval_result_file}
    return {
        'status': 1,
        'dataset_attr': dataset_attr,
        'task_params': task_params,
        'eval_result': eval_result
    }
