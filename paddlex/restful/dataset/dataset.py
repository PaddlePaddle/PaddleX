# copytrue (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ..utils import (set_folder_status, get_folder_status, DatasetStatus,
                     TaskStatus, is_available, DownloadStatus,
                     PretrainedModelStatus, ProjectType)

from threading import Thread
import random
from .utils import copy_directory, get_label_count
import traceback
import shutil
import psutil
import pickle
import os
import os.path as osp
import time
import json
import base64
import cv2
from .. import workspace_pb2 as w


def create_dataset(data, workspace):
    """
    创建dataset
    """
    create_time = time.time()
    time_array = time.localtime(create_time)
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_dataset_id + 1
    if id < 10000:
        did = 'D%04d' % id
    else:
        did = 'D{}'.format(id)
    assert not did in workspace.datasets, "【数据集创建】ID'{}'已经被占用.".format(did)
    path = osp.join(workspace.path, 'datasets', did)
    if osp.exists(path):
        if not osp.isdir(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.makedirs(path)
    set_folder_status(path, DatasetStatus.XEMPTY)
    workspace.max_dataset_id = id
    ds = w.Dataset(
        id=did,
        name=data['name'],
        desc=data['desc'],
        type=data['dataset_type'],
        create_time=create_time,
        path=path)
    workspace.datasets[did].CopyFrom(ds)
    return {'status': 1, 'did': did}


def import_dataset(data, workspace, monitored_processes, load_demo_proc_dict):
    """导入数据集到工作目录，包括数据检查和拷贝
    Args:
        data为dict, key包括
        'did':数据集id，'path': 原数据集目录路径，
        'demo'(可选): 该数据集为demo数据集
    """
    dataset_id = data['did']
    source_path = data['path']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    valid_dataset_type = [
        'classification', 'detection', 'segmentation', 'instance_segmentation',
        'remote_segmentation'
    ]
    assert dataset_type in valid_dataset_type, "无法识别的数据类型{}".format(
        dataset_type)

    from .operate import import_dataset
    process = import_dataset(dataset_id, dataset_type, dataset_path,
                             source_path)
    monitored_processes.put(process.pid)
    if 'demo' in data:
        prj_type = getattr(ProjectType, dataset_type)
        if prj_type not in load_demo_proc_dict:
            load_demo_proc_dict[prj_type] = []
        load_demo_proc_dict[prj_type].append(process)
    return {'status': 1}


def delete_dataset(data, workspace):
    """删除dataset。

    Args:
        data为dict,key包括
        'did'数据集id
    """
    dataset_id = data['did']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    counter = 0
    for key in workspace.projects:
        if workspace.projects[key].did == dataset_id:
            counter += 1
    assert counter == 0, "无法删除数据集,当前仍被{}个项目中使用中,请先删除相关项目".format(counter)
    path = workspace.datasets[dataset_id].path
    if osp.exists(path):
        shutil.rmtree(path)
    del workspace.datasets[dataset_id]
    return {'status': 1}


def get_dataset_status(data, workspace):
    """获取数据集当前状态

    Args:
        data为dict, key包括
        'did':数据集id
    """
    from .operate import get_dataset_status
    dataset_id = data['did']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    dataset_name = workspace.datasets[dataset_id].name
    dataset_desc = workspace.datasets[dataset_id].desc
    dataset_create_time = workspace.datasets[dataset_id].create_time
    status, message = get_dataset_status(dataset_id, dataset_type,
                                         dataset_path)
    dataset_pids = list()
    for key in workspace.projects:
        if dataset_id == workspace.projects[key].did:
            dataset_pids.append(workspace.projects[key].id)

    attr = {
        "type": dataset_type,
        "id": dataset_id,
        "name": dataset_name,
        "path": dataset_path,
        "desc": dataset_desc,
        "create_time": dataset_create_time,
        "pids": dataset_pids
    }
    return {
        'status': 1,
        'id': dataset_id,
        'dataset_status': status.value,
        'message': message,
        'attr': attr
    }


def list_datasets(workspace):
    """
    列出数据集列表，可根据request中的参数进行筛选
    """
    from .operate import get_dataset_status
    dataset_list = list()
    for key in workspace.datasets:
        dataset_type = workspace.datasets[key].type
        dataset_id = workspace.datasets[key].id
        dataset_name = workspace.datasets[key].name
        dataset_path = workspace.datasets[key].path
        dataset_desc = workspace.datasets[key].desc
        dataset_create_time = workspace.datasets[key].create_time
        status, message = get_dataset_status(dataset_id, dataset_type,
                                             dataset_path)
        attr = {
            "type": dataset_type,
            "id": dataset_id,
            "name": dataset_name,
            "path": dataset_path,
            "desc": dataset_desc,
            "create_time": dataset_create_time,
            'dataset_status': status.value,
            'message': message
        }
        dataset_list.append({"id": dataset_id, "attr": attr})
    return {'status': 1, "datasets": dataset_list}


def get_dataset_details(data, workspace):
    """获取数据集详情

    Args:
        data为dict, key包括
        'did':数据集id
    Return:
        details(dict): 'file_info': 全量数据集文件与标签映射表，'label_info': 标签与全量数据集文件映射表，
        'labels': 标签列表，'train_files': 训练集文件列表， 'val_files': 验证集文件列表，
        'test_files': 测试集文件列表
    """
    from .operate import get_dataset_details
    dataset_id = data['did']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_path = workspace.datasets[dataset_id].path
    details = get_dataset_details(dataset_path)
    return {'status': 1, 'details': details}


def split_dataset(data, workspace):
    """将数据集切分为训练集、验证集和测试集

    Args:
        data为dict, key包括
        'did':数据集id, 'val_split': 验证集比例, 'test_split': 测试集比例
    """
    from .operate import split_dataset
    from .operate import get_dataset_details
    dataset_id = data['did']
    assert dataset_id in workspace.datasets, "数据集ID'{}'不存在.".format(dataset_id)
    dataset_type = workspace.datasets[dataset_id].type
    dataset_path = workspace.datasets[dataset_id].path
    val_split = data['val_split']
    test_split = data['test_split']
    split_dataset(dataset_id, dataset_type, dataset_path, val_split,
                  test_split)
    return {'status': 1}


def img_base64(data, workspace=None):
    """将数据集切分为训练集、验证集和测试集

    Args:
        data为dict, key包括
        'path':图片绝对路径
    """
    path = data['path']
    path = '/'.join(path.split('\\'))
    if 'did' in data:
        did = data['did']
        lable_type = workspace.datasets[did].type
        ds_path = workspace.datasets[did].path

        ret = get_dataset_details(data, workspace)
        dataset_details = ret['details']
        ds_label_count = get_label_count(dataset_details['label_info'])
        image_path = 'JPEGImages/' + path.split('/')[-1]
        anno = osp.join(ds_path, dataset_details["file_info"][image_path])

        if lable_type == 'detection':
            from ..project.visualize import plot_det_label
            labels = list(ds_label_count.keys())
            img = plot_det_label(path, anno, labels)
            base64_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode(
            )
            return {'status': 1, 'img_data': base64_str}
        elif lable_type == 'segmentation' or lable_type == 'remote_segmentation':
            from ..project.visualize import plot_seg_label
            im = plot_seg_label(anno)
            img = cv2.imread(path)
            im = cv2.addWeighted(img, 0.5, im, 0.5, 0).astype('uint8')
            base64_str = base64.b64encode(cv2.imencode('.png', im)[1]).decode()
            return {'status': 1, 'img_data': base64_str}
        elif lable_type == 'instance_segmentation':
            labels = list(ds_label_count.keys())
            from ..project.visualize import plot_insseg_label
            img = plot_insseg_label(path, anno, labels)
            base64_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode(
            )
            return {'status': 1, 'img_data': base64_str}
        else:
            raise Exception("数据集类型{}目前暂不支持".format(lable_type))

    with open(path, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        base64_str = str(base64_data, 'utf-8')
    return {'status': 1, 'img_data': base64_str}
