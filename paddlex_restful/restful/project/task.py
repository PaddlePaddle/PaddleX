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

from .. import workspace_pb2 as w
import os
import os.path as osp
import shutil
import time
import pickle
import json
import multiprocessing as mp
import xlwt
import numpy as np
from ..utils import set_folder_status, TaskStatus, get_folder_status, is_available, get_ip, trans_name
from .train.params import ClsParams, DetParams, SegParams


def create_task(data, workspace):
    """根据request创建task。

    Args:
        data为dict,key包括
        'pid'所属项目id, 'train'训练参数。训练参数和数据增强参数以pickle的形式保存
        在任务目录下的params.pkl文件中。 'parent_id'(可选)该裁剪训练任务的父任务,
        'desc'(可选)任务描述。
    """
    create_time = time.time()
    time_array = time.localtime(create_time)
    create_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    id = workspace.max_task_id + 1
    workspace.max_task_id = id
    if id < 10000:
        id = 'T%04d' % id
    else:
        id = 'T{}'.format(id)
    pid = data['pid']
    assert pid in workspace.projects, "【任务创建】项目ID'{}'不存在.".format(pid)
    assert not id in workspace.tasks, "【任务创建】任务ID'{}'已经被占用.".format(id)
    did = workspace.projects[pid].did
    assert did in workspace.datasets, "【任务创建】数据集ID'{}'不存在".format(did)
    path = osp.join(workspace.projects[pid].path, id)
    if not osp.exists(path):
        os.makedirs(path)
    set_folder_status(path, TaskStatus.XINIT)
    data['task_type'] = workspace.projects[pid].type
    data['dataset_path'] = workspace.datasets[did].path
    data['pretrain_weights_download_save_dir'] = osp.join(workspace.path,
                                                          'pretrain')
    #获取参数
    if 'train' in data:
        params_json = json.loads(data['train'])
        if (data['task_type'] == 'classification'):
            params_init = ClsParams()
        if (data['task_type'] == 'detection' or
                data['task_type'] == 'instance_segmentation'):
            params_init = DetParams()
        if (data['task_type'] == 'segmentation' or
                data['task_type'] == 'remote_segmentation'):
            params_init = SegParams()
        params_init.load_from_dict(params_json)
        data['train'] = params_init
    parent_id = ''
    if 'parent_id' in data:
        data['tid'] = data['parent_id']
        parent_id = data['parent_id']
        assert data['parent_id'] in workspace.tasks, "【任务创建】裁剪任务创建失败".format(
            data['parent_id'])
        r = get_task_params(data, workspace)
        train_params = r['train']
        data['train'] = train_params
    desc = ""
    if 'desc' in data:
        desc = data['desc']
    with open(osp.join(path, 'params.pkl'), 'wb') as f:
        pickle.dump(data, f)
    task = w.Task(
        id=id,
        pid=pid,
        path=path,
        create_time=create_time,
        parent_id=parent_id,
        desc=desc)
    workspace.tasks[id].CopyFrom(task)

    with open(os.path.join(path, 'info.pb'), 'wb') as f:
        f.write(task.SerializeToString())

    return {'status': 1, 'tid': id}


def delete_task(data, workspace):
    """删除task。

    Args:
        data为dict,key包括
        'tid'任务id
    """
    task_id = data['tid']
    assert task_id in workspace.tasks, "任务ID'{}'不存在.".format(task_id)
    if osp.exists(workspace.tasks[task_id].path):
        shutil.rmtree(workspace.tasks[task_id].path)
    del workspace.tasks[task_id]
    return {'status': 1}


def get_task_params(data, workspace):
    """根据request获取task的参数。

    Args:
        data为dict,key包括
        'tid'任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "【任务创建】任务ID'{}'不存在.".format(tid)
    path = workspace.tasks[tid].path
    with open(osp.join(path, 'params.pkl'), 'rb') as f:
        task_params = pickle.load(f)
    return {'status': 1, 'train': task_params['train']}


def list_tasks(data, workspace):
    '''列出任务列表，可request的参数进行筛选
    Args:
        data为dict, 包括
        'pid'（可选）所属项目id
    '''
    task_list = list()
    for key in workspace.tasks:
        task_id = workspace.tasks[key].id
        task_name = workspace.tasks[key].name
        task_desc = workspace.tasks[key].desc
        task_pid = workspace.tasks[key].pid
        task_path = workspace.tasks[key].path
        task_create_time = workspace.tasks[key].create_time
        task_type = workspace.projects[task_pid].type
        from .operate import get_task_status
        path = workspace.tasks[task_id].path
        status, message = get_task_status(path)
        if data is not None:
            if "pid" in data:
                if data["pid"] != task_pid:
                    continue
        attr = {
            "id": task_id,
            "name": task_name,
            "desc": task_desc,
            "pid": task_pid,
            "path": task_path,
            "create_time": task_create_time,
            "status": status.value,
            'type': task_type
        }
        task_list.append(attr)
    return {'status': 1, 'tasks': task_list}


def set_task_params(data, workspace):
    """根据request设置task的参数。只有在task是TaskStatus.XINIT状态时才有效

    Args:
        data为dict,key包括
        'tid'任务id, 'train'训练参数. 训练
        参数和数据增强参数以pickle的形式保存在任务目录下的params.pkl文件
        中。
    """
    tid = data['tid']
    train = data['train']
    assert tid in workspace.tasks, "【任务创建】任务ID'{}'不存在.".format(tid)
    path = workspace.tasks[tid].path
    status = get_folder_status(path)
    assert status == TaskStatus.XINIT, "该任务不在初始化阶段，设置参数失败"
    with open(osp.join(path, 'params.pkl'), 'rb') as f:
        task_params = pickle.load(f)
    train_json = json.loads(train)
    task_params['train'].load_from_dict(train_json)
    with open(osp.join(path, 'params.pkl'), 'wb') as f:
        pickle.dump(task_params, f)
    return {'status': 1}


def get_default_params(data, workspace, machine_info):
    from .train.params_v2 import get_params
    from ..dataset.dataset import get_dataset_details
    pid = data['pid']
    assert pid in workspace.projects, "项目ID{}不存在.".format(pid)
    project_type = workspace.projects[pid].type
    did = workspace.projects[pid].did

    result = get_dataset_details({'did': did}, workspace)
    if result['status'] == 1:
        details = result['details']
    else:
        raise Exception("Fail to get dataset details!")
    train_num = len(details['train_files'])
    class_num = len(details['labels'])
    if machine_info['gpu_num'] == 0:
        gpu_num = 0
        per_gpu_memory = 0
        gpu_list = None
    else:
        if 'gpu_list' in data:
            gpu_list = data['gpu_list']
            gpu_num = len(gpu_list)
            per_gpu_memory = None
            for gpu_id in gpu_list:
                if per_gpu_memory is None:
                    per_gpu_memory = machine_info['gpu_free_mem'][gpu_id]
                elif machine_info['gpu_free_mem'][gpu_id] < per_gpu_memory:
                    per_gpu_memory = machine_info['gpu_free_mem'][gpu_id]
        else:
            gpu_num = 1
            per_gpu_memory = machine_info['gpu_free_mem'][0]
            gpu_list = [0]
    params = get_params(data, project_type, train_num, class_num, gpu_num,
                        per_gpu_memory, gpu_list)
    return {"status": 1, "train": params}


def get_task_params(data, workspace):
    """根据request获取task的参数。

    Args:
        data为dict,key包括
        'tid'任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "【任务创建】任务ID'{}'不存在.".format(tid)
    path = workspace.tasks[tid].path
    with open(osp.join(path, 'params.pkl'), 'rb') as f:
        task_params = pickle.load(f)
    return {'status': 1, 'train': task_params['train']}


def get_task_status(data, workspace):
    """ 获取任务状态

    Args:
        data为dict, key包括
        'tid'任务id, 'resume'(可选):获取是否可以恢复训练的状态
    """
    from .operate import get_task_status, get_task_max_saved_epochs
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    status, message = get_task_status(path)
    task_pid = workspace.tasks[tid].pid
    task_type = workspace.projects[task_pid].type
    if 'resume' in data:
        max_saved_epochs = get_task_max_saved_epochs(path)
        params = {'tid': tid}
        results = get_task_params(params, workspace)
        total_epochs = results['train'].num_epochs
        resumable = max_saved_epochs > 0 and max_saved_epochs < total_epochs
        return {
            'status': 1,
            'task_status': status.value,
            'message': message,
            'resumable': resumable,
            'max_saved_epochs': max_saved_epochs,
            'type': task_type
        }

    return {
        'status': 1,
        'task_status': status.value,
        'message': message,
        'type': task_type
    }


def get_train_metrics(data, workspace):
    """ 获取任务日志

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        train_log(dict): 'eta':剩余时间，'train_metrics': 训练指标，'eval_metircs': 评估指标，
        'download_status': 下载模型状态，'eval_done': 是否已保存模型，'train_error': 训练错误原因
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    from ..utils import TrainLogReader
    task_path = workspace.tasks[tid].path
    log_file = osp.join(task_path, 'out.log')
    train_log = TrainLogReader(log_file)
    train_log.update()
    train_log = train_log.__dict__
    return {'status': 1, 'train_log': train_log}


def get_eval_metrics(data, workspace):
    """ 获取任务日志

    Args:
        data为dict, key包括
        'tid'父任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    best_model_path = osp.join(workspace.tasks[tid].path, "output",
                               "best_model", "model.yml")
    import yaml
    f = open(best_model_path, "r", encoding="utf-8")
    eval_metrics = yaml.load(f)['_Attributes']['eval_metrics']
    f.close()
    return {'status': 1, 'eval_metric': eval_metrics}


def get_eval_all_metrics(data, workspace):
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    output_dir = osp.join(workspace.tasks[tid].path, "output")
    epoch_result_dict = dict()
    best_epoch = -1
    best_result = -1
    import yaml
    for file_dir in os.listdir(output_dir):
        if file_dir.startswith("epoch"):
            epoch_dir = osp.join(output_dir, file_dir)
            if osp.exists(osp.join(epoch_dir, ".success")):
                epoch_index = int(file_dir.split('_')[-1])
                yml_file_path = osp.join(epoch_dir, "model.yml")
                f = open(yml_file_path, 'r', encoding='utf-8')
                yml_file = yaml.load(f.read())
                result = yml_file["_Attributes"]["eval_metrics"]
                key = list(result.keys())[0]
                value = result[key]
                if value > best_result:
                    best_result = value
                    best_epoch = epoch_index
                elif value == best_result:
                    if epoch_index < best_epoch:
                        best_epoch = epoch_index
                epoch_result_dict[epoch_index] = value
    return {
        'status': 1,
        'key': key,
        'epoch_result_dict': epoch_result_dict,
        'best_epoch': best_epoch,
        'best_result': best_result
    }


def get_sensitivities_loss_img(data, workspace):
    """ 获取敏感度与模型裁剪率关系图

    Args:
        data为dict, key包括
        'tid'任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    task_path = workspace.tasks[tid].path
    pkl_path = osp.join(task_path, 'prune', 'sensitivities_xy.pkl')
    import pickle
    f = open(pkl_path, 'rb')
    sensitivities_xy = pickle.load(f)
    return {'status': 1, 'sensitivities_loss_img': sensitivities_xy}


def start_train_task(data, workspace, monitored_processes):
    """启动训练任务。

    Args:
        data为dict,key包括
        'tid'任务id, 'eval_metric_loss'（可选）裁剪任务所需的评估loss
    """
    from .operate import train_model
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    if 'eval_metric_loss' in data and \
        data['eval_metric_loss'] is not None:
        # 裁剪任务
        parent_id = workspace.tasks[tid].parent_id
        assert parent_id != "", "任务{}不是裁剪训练任务".format(tid)
        parent_path = workspace.tasks[parent_id].path
        sensitivities_path = osp.join(parent_path, 'prune',
                                      'sensitivities.data')
        eval_metric_loss = data['eval_metric_loss']
        parent_best_model_path = osp.join(parent_path, 'output', 'best_model')
        params_conf_file = osp.join(path, 'params.pkl')
        with open(params_conf_file, 'rb') as f:
            params = pickle.load(f)
        params['train'].sensitivities_path = sensitivities_path
        params['train'].eval_metric_loss = eval_metric_loss
        params['train'].pretrain_weights = parent_best_model_path
        with open(params_conf_file, 'wb') as f:
            pickle.dump(params, f)
    p = train_model(path)
    monitored_processes.put(p.pid)
    return {'status': 1}


def resume_train_task(data, workspace, monitored_processes):
    """恢复训练任务

    Args:
        data为dict, key包括
        'tid'任务id，'epoch'恢复训练的起始轮数
    """
    from .operate import train_model
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    epoch_path = "epoch_" + str(data['epoch'])
    resume_checkpoint_path = osp.join(path, "output", epoch_path)

    params_conf_file = osp.join(path, 'params.pkl')
    with open(params_conf_file, 'rb') as f:
        params = pickle.load(f)
    params['train'].resume_checkpoint = resume_checkpoint_path
    with open(params_conf_file, 'wb') as f:
        pickle.dump(params, f)

    p = train_model(path)
    monitored_processes.put(p.pid)
    return {'status': 1}


def stop_train_task(data, workspace):
    """停止训练任务

    Args:
        data为dict, key包括
        'tid'任务id
    """
    from .operate import stop_train_model
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    stop_train_model(path)
    return {'status': 1}


def start_prune_analysis(data, workspace, monitored_processes):
    """开始模型裁剪分析

    Args:
        data为dict, key包括
        'tid'任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    task_path = workspace.tasks[tid].path
    from .operate import prune_analysis_model
    p = prune_analysis_model(task_path)
    monitored_processes.put(p.pid)
    return {'status': 1}


def get_prune_metrics(data, workspace):
    """ 获取模型裁剪分析日志
    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        prune_log(dict): 'eta':剩余时间，'iters': 模型裁剪总轮数，'current': 当前轮数，
        'progress': 模型裁剪进度
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    from ..utils import PruneLogReader
    task_path = workspace.tasks[tid].path
    log_file = osp.join(task_path, 'prune', 'out.log')
    # assert osp.exists(log_file), "模型裁剪分析任务还未开始，请稍等"
    if not osp.exists(log_file):
        return {'status': 1, 'prune_log': None}
    prune_log = PruneLogReader(log_file)
    prune_log.update()
    prune_log = prune_log.__dict__
    return {'status': 1, 'prune_log': prune_log}


def get_prune_status(data, workspace):
    """ 获取模型裁剪状态

    Args:
        data为dict, key包括
        'tid'任务id
    """
    from .operate import get_prune_status
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    prune_path = osp.join(path, "prune")
    status, message = get_prune_status(prune_path)
    if status is not None:
        status = status.value
    return {'status': 1, 'prune_status': status, 'message': message}


def stop_prune_analysis(data, workspace):
    """停止模型裁剪分析
    Args:
        data为dict, key包括
        'tid'任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    from .operate import stop_prune_analysis
    prune_path = osp.join(workspace.tasks[tid].path, 'prune')
    stop_prune_analysis(prune_path)
    return {'status': 1}


def evaluate_model(data, workspace, monitored_processes):
    """ 模型评估

    Args:
        data为dict, key包括
        'tid'任务id, topk, score_thresh, overlap_thresh这些评估所需参数
    Return:
        None
    """
    from .operate import evaluate_model
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    pid = workspace.tasks[tid].pid
    assert pid in workspace.projects, "项目ID'{}'不存在".format(pid)
    path = workspace.tasks[tid].path
    type = workspace.projects[pid].type
    p = evaluate_model(path, type, data['epoch'], data['topk'],
                       data['score_thresh'], data['overlap_thresh'])
    monitored_processes.put(p.pid)
    return {'status': 1}


def get_evaluate_result(data, workspace):
    """ 获评估结果

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        包含评估指标的dict
    """
    from .operate import get_evaluate_status
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    task_path = workspace.tasks[tid].path
    status, message = get_evaluate_status(task_path)
    if status == TaskStatus.XEVALUATED:
        result_file = osp.join(task_path, 'eval_res.pkl')
        if os.path.exists(result_file):
            result = pickle.load(open(result_file, "rb"))
            return {
                'status': 1,
                'evaluate_status': status,
                'message': "{}评估完成".format(tid),
                'path': result_file,
                'result': result
            }
        else:
            return {
                'status': -1,
                'evaluate_status': status,
                'message': "评估结果丢失，建议重新评估!",
                'result': None
            }
    if status == TaskStatus.XEVALUATEFAIL:
        return {
            'status': -1,
            'evaluate_status': status,
            'message': "评估失败，请重新评估!",
            'result': None
        }
    return {
        'status': 1,
        'evaluate_status': status,
        'message': "{}正在评估中，请稍后!".format(tid),
        'result': None
    }


def import_evaluate_excel(data, result, workspace):
    excel_ret = dict()
    workbook = xlwt.Workbook()
    labels = None
    START_ROW = 0
    sheet = workbook.add_sheet("评估报告")
    if 'label_list' not in result:
        pass
    else:
        labels = result['label_list']
    for k, v in result.items():
        if k == 'label_list':
            continue
        if type(v) == np.ndarray:
            sheet.write(START_ROW + 0, 0, trans_name(k))
            sheet.write(START_ROW + 1, 1, trans_name("Class"))
            if labels is None:
                labels = ["{}".format(x) for x in range(len(v))]
            for i in range(len(labels)):
                sheet.write(START_ROW + 1, 2 + i, labels[i])
                sheet.write(START_ROW + 2 + i, 1, labels[i])
            for i in range(len(labels)):
                for j in range(len(labels)):
                    sheet.write(START_ROW + 2 + i, 2 + j, str(v[i, j]))
            START_ROW = (START_ROW + 4 + len(labels))

        if type(v) == dict:
            sheet.write(START_ROW + 0, 0, trans_name(k))
            multi_row = False
            Cols = ["Class"]
            for k1, v1 in v.items():
                if type(v1) == dict:
                    multi_row = True
                    for sub_k, sub_v in v1.items():
                        Cols.append(sub_k)
                else:
                    Cols.append(k)
                break
            for i in range(len(Cols)):
                sheet.write(START_ROW + 1, 1 + i, trans_name(Cols[i]))

            index = 2
            for k1, v1 in v.items():
                sheet.write(START_ROW + index, 1, k1)
                if multi_row:
                    for sub_k, sub_v in v1.items():
                        sheet.write(START_ROW + index,
                                    Cols.index(sub_k) + 1, "nan"
                                    if (sub_v is None) or sub_v == -1 else
                                    "{:.4f}".format(sub_v))
                else:
                    sheet.write(START_ROW + index, 2, "{}".format(v1))
                index += 1
            START_ROW = (START_ROW + index + 2)
        if type(v) in [float, np.float, np.float32, np.float64, type(None)]:
            front_str = "{}".format(trans_name(k))
            if k == "Acck":
                if "topk" in data:
                    front_str = front_str.format(data["topk"])
                else:
                    front_str = front_str.format(5)
            sheet.write(START_ROW + 0, 0, front_str)
            sheet.write(START_ROW + 1, 1, "{:.4f}".format(v)
                        if v is not None else "nan")
            START_ROW = (START_ROW + 2 + 2)
    tid = data['tid']
    path = workspace.tasks[tid].path
    final_save = os.path.join(path, 'report-task{}.xls'.format(tid))
    workbook.save(final_save)
    excel_ret['status'] = 1
    excel_ret['path'] = final_save
    excel_ret['message'] = "成功导出结果到excel"
    return excel_ret


def get_predict_status(data, workspace):
    from .operate import get_predict_status
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    status, message, predict_num, total_num = get_predict_status(path)
    return {
        'status': 1,
        'predict_status': status.value,
        'message': message,
        'predict_num': predict_num,
        'total_num': total_num
    }


def predict_test_pics(data, workspace, monitored_processes):
    from .operate import predict_test_pics
    tid = data['tid']

    if 'img_list' in data:
        img_list = data['img_list']
    else:
        img_list = list()
    if 'image_data' in data:
        image_data = data['image_data']
    else:
        image_data = None
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    save_dir = data['save_dir'] if 'save_dir' in data else None
    epoch = data['epoch'] if 'epoch' in data else None
    score_thresh = data['score_thresh'] if 'score_thresh' in data else 0.5
    p, save_dir = predict_test_pics(
        path,
        save_dir=save_dir,
        img_list=img_list,
        img_data=image_data,
        score_thresh=score_thresh,
        epoch=epoch)
    monitored_processes.put(p.pid)
    if 'image_data' in data:
        path = osp.join(save_dir, 'predict_result.png')
    else:
        path = None
    return {'status': 1, 'path': path}


def stop_predict_task(data, workspace):
    from .operate import stop_predict_task
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    status, message, predict_num, total_num = stop_predict_task(path)
    return {
        'status': 1,
        'predict_status': status.value,
        'message': message,
        'predict_num': predict_num,
        'total_num': total_num
    }


def get_quant_progress(data, workspace):
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    from ..utils import QuantLogReader
    export_path = osp.join(workspace.tasks[tid].path, "./logs/export")
    log_file = osp.join(export_path, 'out.log')
    quant_log = QuantLogReader(log_file)
    quant_log.update()
    quant_log = quant_log.__dict__
    return {'status': 1, 'quant_log': quant_log}


def get_quant_result(data, workspace):
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    export_path = osp.join(workspace.tasks[tid].path, "./logs/export")
    result_json = osp.join(export_path, 'quant_result.json')
    result = {}
    import json
    if osp.exists(result_json):
        with open(result_json, 'r') as f:
            result = json.load(f)
    return {'status': 1, 'quant_result': result}


def get_export_status(data, workspace):
    """ 获取导出状态

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        目前导出状态.
    """
    from .operate import get_export_status
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    task_path = workspace.tasks[tid].path
    status, message = get_export_status(task_path)
    if status == TaskStatus.XEXPORTED:
        return {
            'status': 1,
            'export_status': status,
            'message': "恭喜您，{}任务模型导出成功！".format(tid)
        }
    if status == TaskStatus.XEXPORTFAIL:
        return {
            'status': -1,
            'export_status': status,
            'message': "{}任务模型导出失败，请重试！".format(tid)
        }
    return {
        'status': 1,
        'export_status': status,
        'message': "{}任务模型导出中，请稍等！".format(tid)
    }


def export_infer_model(data, workspace, monitored_processes):
    """导出部署模型

    Args:
        data为dict，key包括
        'tid'任务id, 'save_dir'导出模型保存路径
    """
    from .operate import export_noquant_model, export_quant_model
    tid = data['tid']
    save_dir = data['save_dir']
    epoch = data['epoch'] if 'epoch' in data else None
    quant = data['quant'] if 'quant' in data else False
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    path = workspace.tasks[tid].path
    if quant:
        p = export_quant_model(path, save_dir, epoch)
    else:
        p = export_noquant_model(path, save_dir, epoch)
    monitored_processes.put(p.pid)
    return {'status': 1, 'save_dir': save_dir}


def export_lite_model(data, workspace):
    """ 导出lite模型

    Args:
        data为dict, key包括
        'tid'任务id, 'save_dir'导出模型保存路径
    """
    from .operate import opt_lite_model
    model_path = data['model_path']
    save_dir = data['save_dir']
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    opt_lite_model(model_path, save_dir)
    if not osp.exists(osp.join(save_dir, "model.nb")):
        if osp.exists(save_dir):
            shutil.rmtree(save_dir)
        return {'status': -1, 'message': "导出为lite模型失败"}
    return {'status': 1, 'message': "完成"}


def stop_export_task(data, workspace):
    """ 停止导出任务

    Args:
        data为dict, key包括
        'tid'任务id
    Return:
        目前导出的状态.
    """
    from .operate import stop_export_task
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    task_path = workspace.tasks[tid].path
    status, message = stop_export_task(task_path)
    return {'status': 1, 'export_status': status.value, 'message': message}


def _open_vdl(logdir, current_port):
    from visualdl.server import app
    app.run(logdir=logdir, host='0.0.0.0', port=current_port)


def open_vdl(data, workspace, current_port, monitored_processes,
             running_boards):
    """打开vdl页面

    Args:
        data为dict,
        'tid' 任务id
    """
    tid = data['tid']
    assert tid in workspace.tasks, "任务ID'{}'不存在".format(tid)
    ip = get_ip()
    if tid in running_boards:
        url = ip + ":{}".format(running_boards[tid][0])
        return {'status': 1, 'url': url}
    task_path = workspace.tasks[tid].path
    logdir = osp.join(task_path, 'output', 'vdl_log')
    assert osp.exists(logdir), "该任务还未正常产生日志文件"
    port_available = is_available(ip, current_port)
    while not port_available:
        current_port += 1
        port_available = is_available(ip, current_port)
        assert current_port <= 8500, "找不到可用的端口"
    p = mp.Process(target=_open_vdl, args=(logdir, current_port))
    p.start()
    monitored_processes.put(p.pid)
    url = ip + ":{}".format(current_port)
    running_boards[tid] = [current_port, p.pid]
    current_port += 1
    total_time = 0
    while True:
        if not is_available(ip, current_port - 1):
            break
        print(current_port)
        time.sleep(0.5)
        total_time += 0.5
        assert total_time <= 8, "VisualDL服务启动超时，请重新尝试打开"

    return {'status': 1, 'url': url}
