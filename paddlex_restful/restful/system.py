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

import sys
import os
import psutil
import platform


def pkill(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except:
        print("Try to kill process {} failed.".format(pid))


def get_system_info(machine_info={}):
    if machine_info:
        return {'status': 1, 'info': machine_info}
    from .utils import get_gpu_info
    gpu_info, message = get_gpu_info()
    cpu_num = os.environ.get('CPU_NUM', 1)
    sysstr = platform.system()
    machine_info['message'] = message
    machine_info['cpu_num'] = cpu_num
    machine_info['gpu_num'] = gpu_info['gpu_num']
    machine_info['sysstr'] = sysstr
    if gpu_info['gpu_num'] > 0:
        machine_info['driver_version'] = gpu_info['driver_version']
        machine_info['gpu_free_mem'] = gpu_info['mem_free']
    return {'status': 1, 'info': machine_info}


def get_gpu_memory_info(machine_info):
    gpu_mem_infos = list()
    if machine_info['gpu_num'] == 0:
        pass
    else:
        from .utils import get_gpu_info
        gpu_info, message = get_gpu_info()
        for i in range(gpu_info['gpu_num']):
            attr = {
                'free': gpu_info['mem_free'][i],
                'used': gpu_info['mem_used'][i],
                'total': gpu_info['mem_total'][i]
            }
            gpu_mem_infos.append(attr)
    return {'status': 1, 'gpu_mem_infos': gpu_mem_infos}


def get_machine_info(data, machine_info):
    path = None
    if "path" in data:
        path = data['path']
    if path in machine_info:
        return {'status': 1, 'info': machine_info}
    from .utils import get_machine_info
    info = get_machine_info(path)
    machine_info = info
    return {'status': 1, 'info': machine_info}


def get_gpu_memory_size(data):
    """获取显存大小

    Args:
        request(comm.Request): 其中request.params为dict, key包括
        'path' 显卡驱动动态链接库路径
    """
    from .utils import PyNvml
    p = PyNvml()
    p.nvml_init(data['path'])
    count = p.nvml_device_get_count()
    gpu_mem_infos = []
    for i in range(count):
        handler = p.nvml_device_get_handle_by_index(i)
        mem = p.nvml_device_get_memory_info(handler)
        attr = {'free': mem.free, 'used': mem.used, 'total': mem.total}
        gpu_mem_infos.append(attr)
    return {'status': 1, 'gpu_mem_infos': gpu_mem_infos}


def exit_system(monitored_processes):
    while not monitored_processes.empty():
        pid = monitored_processes.get(timeout=0.5)
        print("Try to kill process {}".format(pid))
        pkill(pid)
    return {'status': 1}
