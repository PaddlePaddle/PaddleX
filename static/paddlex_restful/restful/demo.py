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

import os
import json
from os import path as osp
from .utils import DownloadStatus, DatasetStatus, ProjectType, get_folder_status
from .project.train.params import PARAMS_CLASS_LIST
from .utils import CustomEncoder

prj_type_list = [
    'classification', 'detection', 'segmentation', 'instance_segmentation'
]


def download_demo_dataset(data, workspace, load_demo_proc_dict):
    """下载样例工程

        Args:
            data为dict, key包括
            'prj_type' 样例类型(ProjectType)
        """
    if isinstance(data['prj_type'], str):
        prj_type = ProjectType(prj_type_list.index(data['prj_type']))
    else:
        prj_type = ProjectType(data['prj_type'])
    assert prj_type.value >= 0 and prj_type.value <= 4, "不支持此样例类型的导入(type:{})".format(
        prj_type)
    target_path = osp.join(workspace.path, "demo_datasets")
    if not osp.exists(target_path):
        os.makedirs(target_path)
    from .dataset.operate import download_demo_dataset
    proc = download_demo_dataset(prj_type, target_path)
    if prj_type not in load_demo_proc_dict:
        load_demo_proc_dict[prj_type] = []
    load_demo_proc_dict[prj_type].append(proc)
    return {'status': 1}


def load_demo_project(data, workspace, monitored_processes,
                      load_demo_proj_data_dict, load_demo_proc_dict):
    """导入样例工程

    Args:
        data为dict, key包括
        'prj_type' 样例类型(ProjectType)
    """
    if isinstance(data['prj_type'], str):
        prj_type = ProjectType(prj_type_list.index(data['prj_type']))
    else:
        prj_type = ProjectType(data['prj_type'])
    assert prj_type.value >= 0 and prj_type.value <= 4, "不支持此样例类型的导入(type:{})".format(
        prj_type)

    target_path = osp.join(workspace.path, "demo_datasets")
    assert osp.exists(target_path), "样例数据集暂未下载，无法导入样例工程"
    target_path = osp.join(target_path, prj_type.name)
    assert osp.exists(target_path), "样例{}数据集暂未下载，无法导入样例工程".format(
        prj_type.name)

    status = get_folder_status(target_path)
    assert status == DownloadStatus.XDDECOMPRESSED, "样例{}数据集暂未解压，无法导入样例工程".format(
        prj_type.name)

    from .dataset.operate import dataset_url_list
    url = dataset_url_list[prj_type.value]
    fname = osp.split(url)[-1]
    for suffix in ['tar', 'tgz', 'zip']:
        pos = fname.find(suffix)
        if pos >= 2:
            fname = fname[0:pos - 1]
            break
    source_dataset_path = osp.join(target_path, fname)
    params_path = osp.join(target_path, fname, fname + "_params.json")
    params = {}
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    dataset_params = params['dataset_info']
    proj_params = params['project_info']
    train_params = params['train_params']

    # 判断数据集、项目名称是否已存在
    dataset_name = dataset_params['name']
    project_name = proj_params['name']
    for id in workspace.datasets:
        if dataset_name == workspace.datasets[id].name:
            return {'status': 1, 'loading_status': 'dataset already exists'}

    for id in workspace.projects:
        if project_name == workspace.projects[id].name:
            return {'status': 1, 'loading_status': 'project already exists'}

    # 创建数据集
    from .dataset.dataset import create_dataset
    results = create_dataset(dataset_params, workspace)
    dataset_id = results['did']

    # 导入数据集
    from .dataset.dataset import import_dataset
    data = {'did': dataset_id, 'path': source_dataset_path}
    import_dataset(data, workspace, monitored_processes, load_demo_proc_dict)

    # 创建项目
    from .project.project import create_project
    results = create_project(proj_params, workspace)

    pid = results['pid']
    # 绑定数据集
    from .workspace import set_attr
    attr_dict = {'did': dataset_id}
    params = {'struct': 'project', 'id': pid, 'attr_dict': attr_dict}
    set_attr(params, workspace)
    # 创建任务
    task_params = PARAMS_CLASS_LIST[prj_type.value]()
    for k, v in train_params.items():
        if hasattr(task_params, k):
            setattr(task_params, k, v)
    task_params = CustomEncoder().encode(task_params)
    from .project.task import create_task
    params = {'pid': pid, 'train': task_params}
    create_task(params, workspace)
    load_demo_proj_data_dict[prj_type] = (pid, dataset_id)
    return {'status': 1, 'did': dataset_id, 'pid': pid}


def get_download_demo_progress(data, workspace):
    """查询样例工程的下载进度

    Args:
        data为dict, key包括
        'prj_type' 样例类型(ProjectType)
    """
    if isinstance(data['prj_type'], str):
        target_path = osp.join(workspace.path, "demo_datasets",
                               data['prj_type'])
    else:
        prj_type = ProjectType(data['prj_type'])
        target_path = osp.join(workspace.path, "demo_datasets", prj_type.name)
    status, message = get_folder_status(target_path, True)
    if status == DownloadStatus.XDDOWNLOADING:
        if isinstance(data['prj_type'], str):
            from .dataset.operate import dataset_url_dict
            url = dataset_url_dict[data['prj_type']]
        else:
            from .dataset.operate import dataset_url_list
            url = dataset_url_list[prj_type.value]
        fname = osp.split(url)[-1] + "_tmp"
        fullname = osp.join(target_path, fname)
        total_size = int(message)
        download_size = osp.getsize(fullname)
        message = download_size * 100 / total_size
    if status is not None:
        attr = {'status': status.value, 'progress': message}
    else:
        attr = {'status': status, 'progress': message}
    return {'status': 1, 'attr': attr}


def stop_import_demo(data, workspace, load_demo_proc_dict,
                     load_demo_proj_data_dict):
    """停止样例工程的导入进度

    Args:
        request(comm.Request): 其中request.params为dict, key包括
        'prj_type' 样例类型(ProjectType)
    """
    if isinstance(data['prj_type'], str):
        prj_type = ProjectType(prj_type_list.index(data['prj_type']))
    else:
        prj_type = ProjectType(data['prj_type'])
    for proc in load_demo_proc_dict[prj_type]:
        if proc.is_alive():
            proc.terminate()
    # 只删除未完成导入的样例项目
    if prj_type in load_demo_proj_data_dict:
        pid, did = load_demo_proj_data_dict[prj_type]
        params = {'did': did}
        from .dataset.dataset import get_dataset_status
        results = get_dataset_status(params, workspace)
        dataset_status = DatasetStatus(results['dataset_status'])
        if dataset_status not in [
                DatasetStatus.XCOPYDONE, DatasetStatus.XSPLITED
        ]:
            params = {'pid': pid}
            from .project.project import delete_project
            delete_project(params, workspace)
            from .dataset.dataset import delete_dataset
            params = {'did': did}
            delete_dataset(params, workspace)
    return {'status': 1}
