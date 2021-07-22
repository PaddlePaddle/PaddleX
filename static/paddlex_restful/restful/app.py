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

from flask import Flask, request, render_template, send_from_directory, jsonify, session, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS
import argparse
from os import path as osp
import os
import time
import json
import sys
import multiprocessing as mp
from . import workspace_pb2 as w
from .utils import CustomEncoder, ShareData, is_pic, get_logger, TaskStatus, get_ip
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
SD = ShareData()


def init(dirname, logger):
    #初始化工作空间
    from .workspace import init_workspace
    from .system import get_system_info
    SD.workspace = w.Workspace(path=dirname)
    init_workspace(SD.workspace, dirname, logger)
    SD.workspace_dir = dirname
    get_system_info(SD.machine_info)


@app.errorhandler(Exception)
def handle_exception(e):
    ret = {"status": -1, 'message': repr(e)}
    return ret


@app.route('/workspace', methods=['GET', 'PUT'])
def workspace():
    """
    methods=='GET':获取工作目录中项目、数据集、任务的属性
        Args：
            struct(str)：结构类型，可以是'dataset', 'project'或'task'，
            id(str)：结构类型对应的id
            attr_list(list):需要获取的属性的列表
        Return：
            attr(dict):key为属性，value为属性的值，
            status
    methods=='PUT':修改工作目录中项目、数据集、任务的属性
        Args:
            struct(str)：结构类型，可以是'dataset', 'project'或'task'，
            id(str)：结构类型对应的id
            attr_dict(dict):key:需要修改的属性，value：需要修改属性的值
        Return:
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if data:
            from .workspace import get_attr
            ret = get_attr(data, SD.workspace)
            return ret
        return {'status': 1, 'dirname': SD.workspace_dir}
    if request.method == 'PUT':
        from .workspace import set_attr
        ret = set_attr(data, SD.workspace)
        return ret


@app.route('/dataset', methods=['GET', 'POST', 'PUT', 'DELETE'])
def dataset():
    """
    methods=='GET':获取所有数据集或者单个数据集的信息
        Args：
            did(str, optional):数据集id（可选），如果存在就返回数据集id对应数据集的信息
        Ruturn：
            status
            if 'did' in Args：
                id(str):数据集id，
                dataset_status(int):数据集状态(DatasetStatus)枚举变量的值
                message(str)：数据集状态信息
                attr(dict):数据集属性
            else:
                datasets(list):所有数据集属性的列表

    methods=='POST':创建一个新的数据集
        Args:
            name(str):数据集名字
            desc(str):数据集描述
            dataset_type(str):数据集类型，可以是['classification', 'detection', 'segmentation','instance_segmentation','remote_segmentation']
        Return:
            did(str):数据集id
            status

    methods=='PUT':异步，向数据集导入数据，支持分类、检测、语义分割、实例分割、摇杆分割数据集类型
        Args：
            did(str)：数据集id
            path(str)：数据集路径
        Return:
            status

    methods=='DELETE':删除已有的某个数据集
        Args:
            did(str)：数据集id
        Return:
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if 'did' in data:
            from .dataset.dataset import get_dataset_status
            ret = get_dataset_status(data, SD.workspace)
            return ret
        from .dataset.dataset import list_datasets
        ret = list_datasets(SD.workspace)
        return ret
    if request.method == 'POST':
        from .dataset.dataset import create_dataset
        ret = create_dataset(data, SD.workspace)
        return ret
    if request.method == 'PUT':
        from .dataset.dataset import import_dataset
        ret = import_dataset(data, SD.workspace, SD.monitored_processes,
                             SD.load_demo_proc_dict)
        return ret

    if request.method == 'DELETE':
        from .dataset.dataset import delete_dataset
        ret = delete_dataset(data, SD.workspace)
        return ret


@app.route('/dataset/details', methods=['GET'])
def dataset_details():
    """
    methods=='GET':获取某个数据集的详细信息
        Args：
            did(str)：数据集id
        Return:
            details(dict)：数据集详细信息,
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .dataset.dataset import get_dataset_details
        ret = get_dataset_details(data, SD.workspace)
        return ret


@app.route('/dataset/split', methods=['PUT'])
def dataset_split():
    """
    Args：
        did(str)：数据集id
        val_split(float): 验证集比例
        test_split(float): 测试集比例
    Return:
        status
    """
    data = request.get_json()
    if request.method == 'PUT':
        from .dataset.dataset import split_dataset
        ret = split_dataset(data, SD.workspace)
        return ret


@app.route('/dataset/image', methods=['GET'])
def dataset_img_base64():
    """
    Args:
        GET: 获取图片base64数据，参数：'path' 图片绝对路径
    """
    data = request.get_json()
    if request.method == 'GET':
        from .dataset.dataset import img_base64
        ret = img_base64(data)
        return ret


@app.route('/dataset/file', methods=['GET'])
def get_image_file():
    """
    Args:
        GET: 获取文件数据，参数：'path' 文件绝对路径
    """
    data = request.get_json()
    if request.method == 'GET':
        ret = data['path']
        return send_file(ret)


@app.route('/dataset/npy', methods=['GET'])
def get_npyfile():
    """
    Args:
        GET: 获取文件数据，参数：'path' npy文件绝对路径
    """
    data = request.get_json()
    if request.method == 'GET':
        npy = np.load(data['path'], allow_pickle=True).tolist()
        npy['gt_bbox'] = npy['gt_bbox'].tolist()
        return npy


@app.route('/file', methods=['GET'])
def get_file():
    """
    Args：
        path'(str):文件在服务端的路径
	Return:
		#数据为图片
		img_data(str)： base64图片数据
		status
		#数据为xml文件
		ret:数据流
		#数据为log文件
		ret:json数据
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        path = data['path']
        if not os.path.exists(path):
            return {'status': -1}
        if is_pic(path):
            from .dataset.dataset import img_base64
            ret = img_base64(data, SD.workspace)
            return ret
        file_type = path[(path.rfind('.') + 1):]
        if file_type in ['xml', 'npy', 'log']:
            return send_file(path)
        else:
            pass


@app.route('/project', methods=['GET', 'POST', 'DELETE'])
def project():
    """
    methods=='GET':获取指定项目id的信息
        Args:
            'id'(str, optional):项目id，可选，如果存在就返回项目id对应项目的信息
        Return:
            status,
            if 'id' in Args：
                attr(dict):项目属性
            else:
                projects(list):所有项目属性

    methods=='POST':创建一个项目
        Args:
            name(str): 项目名
            desc(str)：项目描述
            project_type(str)：项目类型
        Return:
            pid(str):项目id
            status

    methods=='DELETE':删除一个项目，以及项目相关的task
        Args：
            pid(str)：项目id
        Return:
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .project.project import list_projects
        from .project.project import get_project
        if 'id' in data:
            ret = get_project(data, SD.workspace)
            return ret
        ret = list_projects(SD.workspace)
        return ret
    if request.method == 'POST':
        from .project.project import create_project
        ret = create_project(data, SD.workspace)
        return ret
    if request.method == 'DELETE':
        from .project.project import delete_project
        ret = delete_project(data, SD.workspace)
        return ret


@app.route('/project/task', methods=['GET', 'POST', 'DELETE'])
def task():
    """
    methods=='GET':#获取某个任务的信息或者所有任务的信息
        Args：
            tid(str, optional)：任务id，可选，若存在即返回id对应任务的信息
            resume(str, optional):获取是否可以恢复训练的状态，可选，需在存在tid的情况下才生效
            pid(str, optional)：项目id，可选，若存在即返回该项目id下所有任务信息
        Return:
            status
            if 'tid' in Args：
                task_status(int):任务状态(TaskStatus)枚举变量的值
                message(str)：任务状态信息
                type:任务类型包括{'classification', 'detection', 'segmentation', 'instance_segmentation'}
                resumable(bool):仅Args中存在resume时返回，任务训练是否可以恢复
                max_saved_epochs(int):仅Args中存在resume时返回，当前训练模型保存的最大epoch
            else:
                tasks(list):所有任务属性

    methods=='POST':#创建任务(训练或者裁剪)
        Args:
            pid(str)：项目id
            train(dict)：训练参数
            desc(str, optional):任务描述，可选
            parent_id(str, optional):可选，若存在即表示新建的任务为裁剪任务，parent_id的值为裁剪任务对应的训练任务id
        Return：
            tid(str):任务id
            status

    methods=='DELETE':#删除任务
        Args:
            tid（str）:任务id
        Return:
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if data:
            if 'pid' not in data:
                from .project.task import get_task_status
                ret = get_task_status(data, SD.workspace)
                return ret
        from .project.task import list_tasks
        ret = list_tasks(data, SD.workspace)
        return ret
    if request.method == 'POST':
        from .project.task import create_task
        ret = create_task(data, SD.workspace)
        return ret
    if request.method == 'DELETE':
        from .project.task import delete_task
        ret = delete_task(data, SD.workspace)
        return ret


@app.route('/project/task/params', methods=['GET', 'POST'])
def task_params():
    """
    methods=='GET':#获取任务id对应的参数，或者获取项目默认参数
        Args:
            tid（str, optional）:获取任务对应的参数
            pid(str，optional)：获取项目对应的默认参数
            model_type(str，optional)：pid存在下有效，对应项目下获取指定模型的默认参数
            gpu_list(list，optional):pid存在下有效，默认值为[0]，使用指定的gpu并获取相应的默认参数
        Return:
            train(dict):训练或者裁剪的参数
            status

    methods=='POST':#设置任务参数，将前端用户设置训练参数dict保存在后端的pkl文件中
        Args:
            tid(str):任务id
            train(dict)：训练参数
        Return:
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if 'tid' in data:
            from .project.task import get_task_params
            ret = get_task_params(data, SD.workspace)
            ret['train'] = CustomEncoder().encode(ret['train'])
            ret['train'] = json.loads(ret['train'])
            return ret
        if 'pid' in data:
            from .project.task import get_default_params
            ret = get_default_params(data, SD.workspace, SD.machine_info)
            return ret
    if request.method == 'POST':
        from .project.task import set_task_params
        ret = set_task_params(data, SD.workspace)
        return ret


@app.route('/project/task/metrics', methods=['GET'])
def task_metrics():
    """
    methods=='GET':#获取日志数据
        Args:
            tid(str):任务id
            type(str):可以获取日志的类型，[train,eval,sensitivities,prune]，包括训练，评估，敏感度与模型裁剪率关系图，裁剪的日志
        Return:
            status
            if type == 'train':
                train_log(dict): 训练日志
            elif type == 'eval':
                eval_metrics(dict): 评估结果
            elif type == 'sensitivities':
                sensitivities_loss_img(dict): 敏感度与模型裁剪率关系图
            elif type == 'prune':
                prune_log(dict):裁剪日志
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if data['type'] == 'train':
            from .project.task import get_train_metrics
            ret = get_train_metrics(data, SD.workspace)
            return ret
        if data['type'] == 'eval':
            from .project.task import get_eval_metrics
            ret = get_eval_metrics(data, SD.workspace)
            return ret
        if data['type'] == 'eval_all':
            from .project.task import get_eval_all_metrics
            ret = get_eval_all_metrics(data, SD.workspace)
            return ret
        if data['type'] == 'sensitivities':
            from .project.task import get_sensitivities_loss_img
            ret = get_sensitivities_loss_img(data, SD.workspace)
            return ret
        if data['type'] == 'prune':
            from .project.task import get_prune_metrics
            ret = get_prune_metrics(data, SD.workspace)
            return ret


@app.route('/project/task/train', methods=['POST', 'PUT'])
def task_train():
    """
    methods=='POST':#异步，启动训练或者裁剪任务
        Args：
            tid(str):任务id
            eval_metric_loss(int，optional):可选，裁剪任务时可用，裁剪任务所需的评估loss
        Return:
            status

    methods=='PUT':#改变任务训练的状态，即终止训练或者恢复训练
        Args：
            tid(str)：任务id
            act(str)：[stop,resume]暂停或者恢复
            epoch(int)：(resume下可以设置)恢复训练的起始轮数
        Return:
            status
    """
    data = request.get_json()
    if request.method == 'POST':
        from .project.task import start_train_task
        ret = start_train_task(data, SD.workspace, SD.monitored_processes)
        return ret
    if request.method == 'PUT':
        if data['act'] == 'resume':
            from .project.task import resume_train_task
            ret = resume_train_task(data, SD.workspace, SD.monitored_processes)
            return ret
        if data['act'] == 'stop':
            from .project.task import stop_train_task
            ret = stop_train_task(data, SD.workspace)
            return ret


@app.route('/project/task/train/file', methods=['GET'])
def log_file():
    data = request.get_json()
    if request.method == 'GET':
        path = data['path']
        if not os.path.exists(path):
            return {'status': -1}
        logs = open(path, encoding='utf-8').readlines()
        if len(logs) < 50:
            return {'status': 1, 'log': logs}
        else:
            logs = logs[-50:]
            return {'status': 1, 'log': logs}


@app.route('/project/task/prune', methods=['GET', 'POST', 'PUT'])
def task_prune():
    """
    methods=='GET':#获取裁剪任务的状态
        Args：
            tid(str):任务id
        Return:
            prune_status(int): 裁剪任务状态(PruneStatus)枚举变量的值
            status

    methods=='POST':#异步，创建一个裁剪分析，对于启动裁剪任务前需要先启动裁剪分析
        Args：
            tid(str):任务id
        Return:
            status

    methods=='PUT':#改变裁剪分析任务的状态
        Args：
            tid(str):任务id
            act(str):[stop],目前仅支持停止一个裁剪分析任务
        Return
            status
    """
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .project.task import get_prune_status
        ret = get_prune_status(data, SD.workspace)
        return ret
    if request.method == 'POST':
        from .project.task import start_prune_analysis
        ret = start_prune_analysis(data, SD.workspace, SD.monitored_processes)
        return ret
    if request.method == 'PUT':
        if data['act'] == 'stop':
            from .project.task import stop_prune_analysis
            ret = stop_prune_analysis(data, SD.workspace)
        return ret


@app.route('/project/task/evaluate', methods=['GET', 'POST'])
def task_evaluate():
    '''
    methods=='GET':#获取模型评估的结果
        Args：
            tid(str):任务id
        Return:
            evaluate_status(int): 任务状态(TaskStatus)枚举变量的值
            message(str)：描述评估任务的信息
            result(dict)：如果评估成功，返回评估结果的dict，否则为None
            status

    methods=='POST':#异步，创建一个评估任务
        Args：
            tid(str):任务id
            epoch(int,optional):需要评估的epoch，如果为None则会评估训练时指标最好的epoch
            topk(int,optional):分类任务topk指标,如果为None默认输入为5
            score_thresh(float):检测任务类别的score threshhold值，如果为None默认输入为0.5
            overlap_thresh(float):实例分割任务IOU threshhold值，如果为None默认输入为0.3
        Return:
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .project.task import get_evaluate_result
        ret = get_evaluate_result(data, SD.workspace)
        if ret['evaluate_status'] == TaskStatus.XEVALUATED and ret[
                'result'] is not None:
            if 'Confusion_Matrix' in ret['result']:
                ret['result']['Confusion_Matrix'] = ret['result'][
                    'Confusion_Matrix'].tolist()
            ret['result'] = CustomEncoder().encode(ret['result'])
            ret['result'] = json.loads(ret['result'])
        ret['evaluate_status'] = ret['evaluate_status'].value
        return ret
    if request.method == 'POST':
        from .project.task import evaluate_model
        ret = evaluate_model(data, SD.workspace, SD.monitored_processes)
        return ret


@app.route('/project/task/evaluate/file', methods=['GET'])
def task_evaluate_file():
    data = request.get_json()
    if request.method == 'GET':
        if 'path' in data:
            ret = data['path']
            return send_file(ret)
        else:
            from .project.task import get_evaluate_result
            from .project.task import import_evaluate_excel
            ret = get_evaluate_result(data, SD.workspace)
            if ret['evaluate_status'] == TaskStatus.XEVALUATED and ret[
                    'result'] is not None:
                result = ret['result']
                excel_ret = dict()
                excel_ret = import_evaluate_excel(data, result, SD.workspace)
                return excel_ret
            else:
                excel_ret = dict()
                excel_ret['path'] = None
                excel_ret['status'] = -1
                excel_ret['message'] = "评估尚未完成或评估失败"
                return excel_ret


@app.route('/project/task/predict', methods=['GET', 'POST', 'PUT'])
def task_predict():
    '''
    methods=='GET':#获取预测状态
        Args：
            tid(str):任务id
        Return:
            predict_status(int): 预测任务状态(PredictStatus)枚举变量的值
            message(str): 预测信息
            status

    methods=='POST':#创建预测任务，目前仅支持单张图片的预测
        Args:
            tid(str):任务id
            image_data(str):base64编码的image数据
            score_thresh(float，optional):可选，检测任务时有效，检测类别的score threashold值默认是0.5
            epoch(int,float，optional):可选，选择需要做预测的ephoch，默认为评估指标最好的那一个epoch
        Return:
            path(str):服务器上保存预测结果图片的路径
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .project.task import get_predict_status
        ret = get_predict_status(data, SD.workspace)
        return ret
    if request.method == 'POST':
        from .project.task import predict_test_pics
        ret = predict_test_pics(data, SD.workspace, SD.monitored_processes)
        if 'img_list' in data:
            del ret['path']
            return ret
        return ret
    if request.method == 'PUT':
        from .project.task import stop_predict_task
        ret = stop_predict_task(data, SD.workspace)
        return ret


@app.route('/project/task/export', methods=['GET', 'POST', 'PUT'])
def task_export():
    '''
    methods=='GET':#获取导出模型的状态
        Args：
            tid(str):任务id
            quant(str,optional)可选，[log,result]，导出量模型导出状态，若值为log则返回量化的日志；若值为result则返回量化的结果
        Return:
            status
            if quant == 'log':
                quant_log(dict):量化日志
            if quant == 'result'
                quant_result(dict):量化结果
            if quant not in Args：
                export_status(int):模型导出状态(PredictStatus)枚举变量的值
                message(str):模型导出提示信息

    methods=='POST':#导出inference模型或者导出lite模型
        Args:
            tid(str):任务id
            type(str):保存模型的类别[infer,lite]，支持inference模型导出和lite的模型导出
            save_dir(str):保存模型的路径
            epoch(str,optional)可选，指定导出的epoch数默认为评估效果最好的epoch
            quant(bool,optional)可选，type为infer有效，是否导出量化后的模型，默认为False
            model_path(str,optional)可选，type为lite时有效，inference模型的地址
        Return:
            status
            if type == 'infer':
                save_dir:模型保存路径
            if type == 'lite':
                message:模型保存信息

    methods=='PUT':#停止导出模型
        Args:
            tid(str):任务id
        Return:
            export_status(int):模型导出状态(PredictStatus)枚举变量的值
            message(str):停止模型导出提示信息
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if 'quant' in data:
            if data['quant'] == 'log':
                from .project.task import get_quant_progress
                ret = get_quant_progress(data, SD.workspace)
                return ret
            if data['quant'] == 'result':
                from .project.task import get_quant_result
                ret = get_quant_result(data, SD.workspace)
                return ret
        from .project.task import get_export_status
        ret = get_export_status(data, SD.workspace)
        ret['export_status'] = ret['export_status'].value
        return ret
    if request.method == 'POST':
        if data['type'] == 'infer':
            from .project.task import export_infer_model
            ret = export_infer_model(data, SD.workspace,
                                     SD.monitored_processes)
            return ret
        if data['type'] == 'lite':
            from .project.task import export_lite_model
            ret = export_lite_model(data, SD.workspace)
            return ret
    if request.method == 'PUT':
        from .project.task import stop_export_task
        stop_export_task(data, SD.workspace)
        return ret


@app.route('/project/task/vdl', methods=['GET'])
def task_vdl():
    '''
    methods=='GET':#打开某个任务的可视化分析工具(VisualDL)
        Args:
            tid(str):任务id
        Return:
            url(str):vdl地址
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .project.task import open_vdl
        ret = open_vdl(data, SD.workspace, SD.current_port,
                       SD.monitored_processes, SD.running_boards)
        return ret


@app.route('/system', methods=['GET', 'DELETE'])
def system():
    '''
    methods=='GET':#获取系统GPU、CPU信息
        Args:
            type(str):[machine_info,gpu_memory_size]选择需要获取的系统信息
        Return:
            status
            if type=='machine_info'
                info(dict):服务端信息
            if type=='gpu_memory_size'
                gpu_mem_infos(list):GPU内存信息
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if data['type'] == 'machine_info':
            '''if 'path' not in data:
                data['path'] = None
            from .system import get_machine_info
            ret = get_machine_info(data, SD.machine_info)'''
            from .system import get_system_info
            ret = get_system_info(SD.machine_info)
            return ret

        if data['type'] == 'gpu_memory_size':
            #from .system import get_gpu_memory_size
            from .system import get_gpu_memory_info
            ret = get_gpu_memory_info(SD.machine_info)
            return ret
    if request.method == 'DELETE':
        from .system import exit_system
        ret = exit_system(SD.monitored_processes)
        return ret


@app.route('/demo', methods=['GET', 'POST', 'PUT'])
def demo():
    '''
    methods=='GET':#获取demo下载进度
        Args:
            prj_type(int):项目类型ProjectType枚举变量的int值
        Return:
            status
            attr(dict):demo下载信息

    methods=='POST':#下载或创建demo工程
        Args:
            type(str):{download,load}下载或者创建样例
            prj_type(int):项目类型ProjectType枚举变量的int值
        Return:
            status
            if type=='load':
                did:数据集id
                pid:项目id

    methods=='PUT':#停止下载或创建demo工程
        Args:
            prj_type(int):项目类型ProjectType枚举变量的int值
        Return:
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        from .demo import get_download_demo_progress
        ret = get_download_demo_progress(data, SD.workspace)
        return ret
    if request.method == 'POST':
        if data['type'] == 'download':
            from .demo import download_demo_dataset
            ret = download_demo_dataset(data, SD.workspace,
                                        SD.load_demo_proc_dict)
            return ret
        if data['type'] == 'load':
            from .demo import load_demo_project
            ret = load_demo_project(data, SD.workspace, SD.monitored_processes,
                                    SD.load_demo_proj_data_dict,
                                    SD.load_demo_proc_dict)
            return ret
    if request.method == 'PUT':
        from .demo import stop_import_demo
        ret = stop_import_demo(data, SD.workspace, SD.load_demo_proc_dict,
                               SD.load_demo_proj_data_dict)
        return ret


@app.route('/model', methods=['GET', 'POST', 'DELETE'])
def model():
    '''
    methods=='GET':#获取一个或者所有模型的信息
        Args:
            mid(str,optional)可选，若存在则返回某个模型的信息
            type(str,optional)可选，[pretrained,exported].若存在则返回对应类型下所有的模型信息
        Return:
            status
            if mid in Args:
                dataset_attr(dict):数据集属性
                task_params(dict):模型训练参数
                eval_result(dict):模型评估结果
            if type in Args and type == 'pretrained':
                pretrained_models(list):所有预训练模型信息
            if type in Args and type == 'exported':
                exported_models(list):所有inference模型的信息

    methods=='POST':#创建一个模型
        Args:
            pid(str):项目id
            tid(str):任务id
            name(str):模型名字
            type(str):创建模型的类型,[pretrained,exported],pretrained代表创建预训练模型、exported代表创建inference或者lite模型
            source_path(str):仅type为pretrained时有效，训练好的模型的路径
            path(str):仅type为exported时有效，inference或者lite模型的路径
            exported_type(int):0为inference模型，1为lite模型
            eval_results(dict，optional):可选，仅type为pretrained时有效，模型评估的指标
        Return:
            status
            if type == 'pretrained':
                pmid(str):预训练模型id
            if type == 'exported':
                emid(str):inference模型id

    methods=='DELETE':删除一个模型
        Args:
            type(str):删除模型的类型，[pretrained,exported]，pretrained代表创建预训练模型、exported代表创建inference或者lite模型
            if type='pretrained':
                pmid:预训练模型id
            if type='exported':
                emid:inference或者lite模型id
        Return:
            status
    '''
    data = request.get_json()
    if data is None:
        data = request.args
    if request.method == 'GET':
        if 'type' in data:
            if data['type'] == 'pretrained':
                from .model import list_pretrained_models
                ret = list_pretrained_models(SD.workspace)
                return ret
            if data['type'] == 'exported':
                from .model import list_exported_models
                ret = list_exported_models(SD.workspace)
                return ret
        from .model import get_model_details
        ret = get_model_details(data, SD.workspace)
        ret['eval_result']['Confusion_Matrix'] = ret['eval_result'][
            'Confusion_Matrix'].tolist()
        ret['eval_result'] = CustomEncoder().encode(ret['eval_result'])
        ret['task_params'] = CustomEncoder().encode(ret['task_params'])
        return ret
    if request.method == 'POST':
        if data['type'] == 'pretrained':
            if 'eval_results' in data:
                data['eval_results']['Confusion_Matrix'] = np.array(data[
                    'eval_results']['Confusion_Matrix'])
            from .model import create_pretrained_model
            ret = create_pretrained_model(data, SD.workspace,
                                          SD.monitored_processes)
            return ret
        if data['type'] == 'exported':
            from .model import create_exported_model
            ret = create_exported_model(data, SD.workspace)
            return ret
    if request.method == 'DELETE':
        if data['type'] == 'pretrained':
            from .model import delete_pretrained_model
            ret = delete_pretrained_model(data, SD.workspace)
            return ret
        if data['type'] == 'exported':
            from .model import delete_exported_model
            ret = delete_exported_model(data, SD.workspace)
            return ret


@app.route('/model/file', methods=['GET'])
def model_file():
    data = request.get_json()
    if request.method == 'GET':
        ret = data['path']
        return send_file(ret)


@app.route('/', methods=['GET'])
def gui():
    if request.method == 'GET':
        file_path = osp.join(
            osp.dirname(__file__), 'templates', 'paddlex_restful_demo.html')
        ip = get_ip()
        url = 'var str_srv_url = "http://' + ip + ':' + str(SD.port) + '";'
        f = open(file_path, 'r+')
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '0.0.0.0:8080' in line:
                lines[i] = url
                break
        f.close()
        f = open(file_path, 'w+')
        f.writelines(lines)
        f.close()
        return render_template('/paddlex_restful_demo.html')


def run(port, workspace_dir):
    if workspace_dir is None:
        user_home = os.path.expanduser('~')
        dirname = osp.join(user_home, "paddlex_workspace")
    else:
        dirname = workspace_dir
    if not osp.exists(dirname):
        os.makedirs(dirname)
    else:
        if not osp.isdir(dirname):
            os.remove(dirname)
            os.makedirs(dirname)
    logger = get_logger(osp.join(dirname, "mcessages.log"))
    init(dirname, logger)
    SD.port = port
    ip = get_ip()
    url = ip + ':' + str(port)
    try:
        logger.info("RESTful服务启动成功后，您可以在浏览器打开 {} 使用WEB版本GUI".format(url))
        app.run(host='0.0.0.0', port=port, threaded=True)
    except:
        print("服务启动不成功，请确保端口号：{}未被防火墙限制".format(port))
