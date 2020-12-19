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

import time
import os
import os.path as osp
from .. import workspace_pb2 as w
import shutil


def create_project(data, workspace):
    """根据request创建project。

    Args:
        data为dict,key包括
        'name'项目名, 'desc'项目描述，'project_type'项目类型和'path'
        项目路径，`path`(可选)，不设置或为None，表示使用默认的生成路径
    """
    create_time = time.time()
    time_array = time.localtime(create_time)
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_project_id + 1
    workspace.max_project_id = id
    if id < 10000:
        id = 'P%04d' % id
    else:
        id = 'P{}'.format(id)
    assert not id in workspace.projects, "【项目创建】ID'{}'已经被占用.".format(id)
    if 'path' not in data:
        path = osp.join(workspace.path, 'projects', id)
    if not osp.exists(path):
        os.makedirs(path)
    pj = w.Project(
        id=id,
        name=data['name'],
        desc=data['desc'],
        type=data['project_type'],
        create_time=create_time,
        path=path)
    workspace.projects[id].CopyFrom(pj)

    with open(os.path.join(path, 'info.pb'), 'wb') as f:
        f.write(pj.SerializeToString())

    return {'status': 1, 'pid': id}


def delete_project(data, workspace):
    """删除project和与project相关的task

    Args:
        data为dict,key包括
        'pid'项目id
    """
    proj_id = data['pid']
    assert proj_id in workspace.projects, "项目ID'{}'不存在.".format(proj_id)

    tids = list()
    for key in workspace.tasks:
        tids.append(key)
    for tid in tids:
        if workspace.tasks[tid].pid == proj_id:
            from .task import delete_task
            data['tid'] = tid
            delete_task(data, workspace)
    if osp.exists(workspace.projects[proj_id].path):
        shutil.rmtree(workspace.projects[proj_id].path)
    del workspace.projects[proj_id]
    return {'status': 1}


def list_projects(workspace):
    '''列出项目列表
    Args:
    '''
    project_list = list()
    for key in workspace.projects:
        project_id = workspace.projects[key].id
        project_name = workspace.projects[key].name
        project_desc = workspace.projects[key].desc
        project_type = workspace.projects[key].type
        project_did = workspace.projects[key].did
        project_path = workspace.projects[key].path
        project_create_time = workspace.projects[key].create_time
        attr = {
            "id": project_id,
            "name": project_name,
            "desc": project_desc,
            "type": project_type,
            "did": project_did,
            "path": project_path,
            "create_time": project_create_time
        }
        project_list.append({"id": project_id, "attr": attr})
    return {'status': 1, 'projects': project_list}


def get_project(data, workspace):
    '''获取项目信息
    Args:
        data为dict，
        'id': 项目id
    '''
    pid = data["id"]
    attr = {}
    assert pid in workspace.projects, "项目ID'{}'不存在".format(pid)
    if pid in workspace.projects:
        project_id = workspace.projects[pid].id
        project_name = workspace.projects[pid].name
        project_desc = workspace.projects[pid].desc
        project_type = workspace.projects[pid].type
        project_did = workspace.projects[pid].did
        project_path = workspace.projects[pid].path
        project_create_time = workspace.projects[pid].create_time
        project_tasks = {}
        from .operate import get_task_status
        for tid in workspace.tasks:
            if workspace.tasks[tid].pid == pid:
                path = workspace.tasks[tid].path
                status, message = get_task_status(path)
                project_tasks[tid] = {'status': status.value}
        attr = {
            "id": project_id,
            "name": project_name,
            "desc": project_desc,
            "type": project_type,
            "did": project_did,
            "path": project_path,
            "create_time": project_create_time,
            "tasks": project_tasks
        }
    return {'status': 1, 'attr': attr}
