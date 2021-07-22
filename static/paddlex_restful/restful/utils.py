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

import psutil
import shutil
import os
import os.path as osp
from enum import Enum
import multiprocessing as mp
from queue import Queue
import time
import threading
from ctypes import CDLL, c_char, c_uint, c_ulonglong
from _ctypes import byref, Structure, POINTER
import platform
import string
import logging
import socket
import logging.handlers
import requests
import json
from json import JSONEncoder


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__


class ShareData():
    workspace = None
    workspace_dir = ""
    has_gpu = True
    monitored_processes = mp.Queue(4096)
    port = 5000
    current_port = 8000
    running_boards = {}
    machine_info = dict()
    load_demo_proc_dict = {}
    load_demo_proj_data_dict = {}


DatasetStatus = Enum(
    'DatasetStatus', ('XEMPTY', 'XCHECKING', 'XCHECKFAIL', 'XCOPYING',
                      'XCOPYDONE', 'XCOPYFAIL', 'XSPLITED'),
    start=0)

TaskStatus = Enum(
    'TaskStatus', ('XUNINIT', 'XINIT', 'XDOWNLOADING', 'XTRAINING',
                   'XTRAINDONE', 'XEVALUATED', 'XEXPORTING', 'XEXPORTED',
                   'XTRAINEXIT', 'XDOWNLOADFAIL', 'XTRAINFAIL', 'XEVALUATING',
                   'XEVALUATEFAIL', 'XEXPORTFAIL', 'XPRUNEING', 'XPRUNETRAIN'),
    start=0)

ProjectType = Enum(
    'ProjectType', ('classification', 'detection', 'segmentation',
                    'instance_segmentation', 'remote_segmentation'),
    start=0)

DownloadStatus = Enum(
    'DownloadStatus',
    ('XDDOWNLOADING', 'XDDOWNLOADFAIL', 'XDDOWNLOADDONE', 'XDDECOMPRESSED'),
    start=0)

PredictStatus = Enum(
    'PredictStatus', ('XPRESTART', 'XPREDONE', 'XPREFAIL'), start=0)

PruneStatus = Enum(
    'PruneStatus', ('XSPRUNESTART', 'XSPRUNEING', 'XSPRUNEDONE', 'XSPRUNEFAIL',
                    'XSPRUNEEXIT'),
    start=0)

PretrainedModelStatus = Enum(
    'PretrainedModelStatus',
    ('XPINIT', 'XPSAVING', 'XPSAVEFAIL', 'XPSAVEDONE'),
    start=0)

ExportedModelType = Enum(
    'ExportedModelType', ('XQUANTMOBILE', 'XPRUNEMOBILE', 'XTRAINMOBILE',
                          'XQUANTSERVER', 'XPRUNESERVER', 'XTRAINSERVER'),
    start=0)

translate_chinese_table = {
    "Confusion_matrix": "各个类别之间的混淆矩阵",
    "Precision": "精准率",
    "Accuracy": "准确率",
    "Recall": "召回率",
    "Class": "类别",
    "Topk": "K取值",
    "Auc": "AUC",
    "Per_ap": "类别平均精准率",
    "Map": "类别平均精准率（AP）的均值（mAP）",
    "Mean_iou": "平均交并比",
    "Mean_acc": "平均准确率",
    "Category_iou": "各类别交并比",
    "Category_acc": "各类别准确率",
    "Ap": "平均精准率",
    "F1": "F1-score",
    "Iou": "交并比"
}

translate_chinese = {
    "Confusion_matrix": "混淆矩阵",
    "Mask_confusion_matrix": "Mask混淆矩阵",
    "Bbox_confusion_matrix": "Bbox混淆矩阵",
    "Precision": "精准率（Precision）",
    "Accuracy": "准确率（Accuracy）",
    "Recall": "召回率（Recall）",
    "Class": "类别（Class）",
    "PRF1": "整体分类评估结果",
    "PRF1_TOPk": "TopK评估结果",
    "Topk": "K取值",
    "AUC": "Area Under Curve",
    "Auc": "Area Under Curve",
    "F1": "F1-score",
    "Iou": "交并比（IoU）",
    "Per_ap": "各类别的平均精准率（AP）",
    "mAP": "平均精准率的均值（mAP）",
    "Mask_mAP": "Mask的平均精准率的均值（mAP）",
    "BBox_mAP": "Bbox的平均精准率的均值（mAP）",
    "Mean_iou": "平均交并比（mIoU）",
    "Mean_acc": "平均准确率（mAcc）",
    "Ap": "平均精准率（Average Precision)",
    "Category_iou": "各类别的交并比（IoU）",
    "Category_acc": "各类别的准确率（Accuracy）",
    "PRAP": "整体检测评估结果",
    "BBox_PRAP": "Bbox评估结果",
    "Mask_PRAP": "Mask评估结果",
    "Overall": "整体平均指标",
    "PRF1_average": "整体平均指标",
    "overall_det": "整体平均指标",
    "PRIoU": "整体平均指标",
    "Acc1": "预测Top1的准确率",
    "Acck": "预测Top{}的准确率"
}

process_pool = Queue(1000)


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_logger(filename):
    flask_logger = logging.getLogger()
    flask_logger.setLevel(level=logging.INFO)
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s:%(message)s'
    format_str = logging.Formatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(level=logging.INFO)
    ch.setFormatter(format_str)
    th = logging.handlers.TimedRotatingFileHandler(
        filename=filename, when='D', backupCount=5, encoding='utf-8')
    th.setFormatter(format_str)
    flask_logger.addHandler(th)
    flask_logger.addHandler(ch)
    return flask_logger


def start_process(target, args):
    global process_pool
    p = mp.Process(target=target, args=args)
    p.start()
    process_pool.put(p)


def pkill(pid):
    """结束进程pid，和与其相关的子进程

    Args:
        pid(int): 进程id
    """
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except:
        print("Try to kill process {} failed.".format(pid))


def set_folder_status(dirname, status, message=""):
    """设置目录状态

    Args:
        dirname(str): 目录路径
        status(DatasetStatus): 状态
        message(str): 需要写到状态文件里的信息
    """
    if not osp.isdir(dirname):
        raise Exception("目录路径{}不存在".format(dirname))
    tmp_file = osp.join(dirname, status.name + '.tmp')
    with open(tmp_file, 'w', encoding='utf-8') as f:
        f.write("{}\n".format(message))
    shutil.move(tmp_file, osp.join(dirname, status.name))
    for status_type in [
            DatasetStatus, TaskStatus, PredictStatus, PruneStatus,
            DownloadStatus, PretrainedModelStatus
    ]:
        for s in status_type:
            if s == status:
                continue
            if osp.exists(osp.join(dirname, s.name)):
                os.remove(osp.join(dirname, s.name))


def get_folder_status(dirname, with_message=False):
    """获取目录状态

    Args:
        dirname(str): 目录路径
        with_message(bool): 是否需要返回状态文件内的信息
    """
    status = None
    closest_time = 0
    message = ''
    for status_type in [
            DatasetStatus, TaskStatus, PredictStatus, PruneStatus,
            DownloadStatus, PretrainedModelStatus
    ]:
        for s in status_type:
            if osp.exists(osp.join(dirname, s.name)):
                modify_time = os.stat(osp.join(dirname, s.name)).st_mtime
                if modify_time > closest_time:
                    closest_time = modify_time
                    status = getattr(status_type, s.name)
                    if with_message:
                        encoding = 'utf-8'
                        try:
                            f = open(
                                osp.join(dirname, s.name),
                                'r',
                                encoding=encoding)
                            message = f.read()
                            f.close()
                        except:
                            try:
                                import chardet
                                f = open(filename, 'rb')
                                data = f.read()
                                f.close()
                                encoding = chardet.detect(data).get('encoding')
                                f = open(
                                    osp.join(dirname, s.name),
                                    'r',
                                    encoding=encoding)
                                message = f.read()
                                f.close()
                            except:
                                pass
    if with_message:
        return status, message
    return status


def _machine_check_proc(queue, path):
    info = dict()
    p = PyNvml()
    gpu_num = 0
    try:
        # import paddle.fluid.core as core
        # gpu_num = core.get_cuda_device_count()
        p.nvml_init(path)
        gpu_num = p.nvml_device_get_count()
        driver_version = bytes.decode(p.nvml_system_get_driver_version())
    except:
        driver_version = "N/A"
    info['gpu_num'] = gpu_num
    info['gpu_free_mem'] = list()
    try:
        for i in range(gpu_num):
            handle = p.nvml_device_get_handle_by_index(i)
            meminfo = p.nvml_device_get_memory_info(handle)
            free_mem = meminfo.free / 1024 / 1024
            info['gpu_free_mem'].append(free_mem)
    except:
        pass

    info['cpu_num'] = os.environ.get('CPU_NUM', 1)
    info['driver_version'] = driver_version
    info['path'] = p.nvml_lib_path
    queue.put(info, timeout=2)


def get_machine_info(path=None):
    queue = mp.Queue(1)
    p = mp.Process(target=_machine_check_proc, args=(queue, path))
    p.start()
    p.join()
    return queue.get(timeout=2)


def download(url, target_path):
    if not osp.exists(target_path):
        os.makedirs(target_path)
    fname = osp.split(url)[-1]
    fullname = osp.join(target_path, fname)
    retry_cnt = 0
    DOWNLOAD_RETRY_LIMIT = 3
    while not (osp.exists(fullname)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            # 设置下载失败
            msg = "Download from {} failed. Retry limit reached".format(url)
            set_folder_status(target_path, DownloadStatus.XDDOWNLOADFAIL, msg)
            raise RuntimeError(msg)
        req = requests.get(url, stream=True)
        if req.status_code != 200:
            msg = "Downloading from {} failed with code {}!".format(
                url, req.status_code)
            set_folder_status(target_path, DownloadStatus.XDDOWNLOADFAIL, msg)
            raise RuntimeError(msg)

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        set_folder_status(target_path, DownloadStatus.XDDOWNLOADING,
                          total_size)

        with open(tmp_fullname, 'wb') as f:
            if total_size:
                download_size = 0
                for chunk in req.iter_content(chunk_size=1024):
                    f.write(chunk)
                    download_size += 1024
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
    set_folder_status(target_path, DownloadStatus.XDDOWNLOADDONE)
    return fullname


def trans_name(key, in_table=False):
    if in_table:
        if key in translate_chinese_table:
            key = "{}".format(translate_chinese_table[key])
        if key.capitalize() in translate_chinese_table:
            key = "{}".format(translate_chinese_table[key.capitalize()])
        return key
    else:
        if key in translate_chinese:
            key = "{}".format(translate_chinese[key])
        if key.capitalize() in translate_chinese:
            key = "{}".format(translate_chinese[key.capitalize()])
        return key
    return key


def is_pic(filename):
    suffixes = {'JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png'}
    suffix = filename.strip().split('.')[-1]
    if suffix not in suffixes:
        return False
    return True


def is_available(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return False
    except:
        return True


def list_files(dirname):
    """ 列出目录下所有文件（包括所属的一级子目录下文件）

    Args:
        dirname: 目录路径
    """

    def filter_file(f):
        if f.startswith('.'):
            return True
        if hasattr(PretrainedModelStatus, f):
            return True
        return False

    all_files = list()
    dirs = list()
    for f in os.listdir(dirname):
        if filter_file(f):
            continue
        if osp.isdir(osp.join(dirname, f)):
            dirs.append(f)
        else:
            all_files.append(f)
    for d in dirs:
        for f in os.listdir(osp.join(dirname, d)):
            if filter_file(f):
                continue
            if osp.isdir(osp.join(dirname, d, f)):
                continue
            all_files.append(osp.join(d, f))
    return all_files


def copy_model_directory(src, dst, files=None, filter_files=[]):
    """从src目录copy文件至dst目录，
           注意:拷贝前会先清空dst中的所有文件

        Args:
            src: 源目录路径
            dst: 目标目录路径
            files: 需要拷贝的文件列表（src的相对路径）
        """
    set_folder_status(dst, PretrainedModelStatus.XPSAVING, os.getpid())
    if files is None:
        files = list_files(src)
    try:
        message = '{} {}'.format(os.getpid(), len(files))
        set_folder_status(dst, PretrainedModelStatus.XPSAVING, message)
        if not osp.samefile(src, dst):
            for i, f in enumerate(files):
                items = osp.split(f)
                if len(items) > 2:
                    continue
                if len(items) == 2:
                    if not osp.isdir(osp.join(dst, items[0])):
                        if osp.exists(osp.join(dst, items[0])):
                            os.remove(osp.join(dst, items[0]))
                        os.makedirs(osp.join(dst, items[0]))
                if f not in filter_files:
                    shutil.copy(osp.join(src, f), osp.join(dst, f))
        set_folder_status(dst, PretrainedModelStatus.XPSAVEDONE)
    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        set_folder_status(dst, PretrainedModelStatus.XPSAVEFAIL, error_info)


def copy_pretrained_model(src, dst):
    p = mp.Process(
        target=copy_model_directory, args=(src, dst, None, ['model.pdopt']))
    p.start()
    return p


def _get_gpu_info(queue):
    gpu_info = dict()
    mem_free = list()
    mem_used = list()
    mem_total = list()
    import pycuda.driver as drv
    from pycuda.tools import clear_context_caches
    drv.init()
    driver_version = drv.get_driver_version()
    gpu_num = drv.Device.count()
    for gpu_id in range(gpu_num):
        dev = drv.Device(gpu_id)
        try:
            context = dev.make_context()
            free, total = drv.mem_get_info()
            context.pop()
            free = free // 1024 // 1024
            total = total // 1024 // 1024
            used = total - free
        except:
            free = 0
            total = 0
            used = 0
        mem_free.append(free)
        mem_used.append(used)
        mem_total.append(total)
    gpu_info['mem_free'] = mem_free
    gpu_info['mem_used'] = mem_used
    gpu_info['mem_total'] = mem_total
    gpu_info['driver_version'] = driver_version
    gpu_info['gpu_num'] = gpu_num
    queue.put(gpu_info)


def get_gpu_info():
    try:
        import pycuda
    except:
        gpu_info = dict()
        message = "未检测到GPU \n 若存在GPU请确保安装pycuda \n 若未安装pycuda请使用'pip install pycuda'来安装"
        gpu_info['gpu_num'] = 0
        return gpu_info, message
    queue = mp.Queue(1)
    p = mp.Process(target=_get_gpu_info, args=(queue, ))
    p.start()
    p.join()
    gpu_info = queue.get(timeout=2)
    if gpu_info['gpu_num'] == 0:
        message = "未检测到GPU"
    else:
        message = "检测到GPU"

    return gpu_info, message


class TrainLogReader(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.eta = None
        self.train_metrics = None
        self.eval_metrics = None
        self.download_status = None
        self.eval_done = False
        self.train_error = None
        self.train_stage = None
        self.running_duration = None

    def update(self):
        if not osp.exists(self.log_file):
            return
        if self.train_stage == "Train Error":
            return
        if self.download_status == "Failed":
            return
        if self.train_stage == "Train Complete":
            return
        logs = open(self.log_file, encoding='utf-8').read().strip().split('\n')
        self.eta = None
        self.train_metrics = None
        self.eval_metrics = None
        if self.download_status != "Done":
            self.download_status = None

        start_time_timestamp = osp.getctime(self.log_file)
        for line in logs[::1]:
            try:
                start_time_str = " ".join(line.split()[0:2])
                start_time_array = time.strptime(start_time_str,
                                                 "%Y-%m-%d %H:%M:%S")
                start_time_timestamp = time.mktime(start_time_array)
                break
            except Exception as e:
                pass
        for line in logs[::-1]:
            if line.count('Train Complete!'):
                self.train_stage = "Train Complete"
            if line.count('Training stop with error!'):
                self.train_error = line
            if self.train_metrics is not None \
                    and self.eval_metrics is not None and self.eval_done and self.eta is not None:
                break
            items = line.strip().split()
            if line.count('Model saved in'):
                self.eval_done = True
            if line.count('download completed'):
                self.download_status = 'Done'
                break
            if line.count('download failed'):
                self.download_status = 'Failed'
                break
            if self.download_status != 'Done':
                if line.count('[DEBUG]\tDownloading'
                              ) and self.download_status is None:
                    self.download_status = dict()
                    if not line.endswith('KB/s'):
                        continue
                    speed = items[-1].strip('KB/s').split('=')[-1]
                    download = items[-2].strip('M, ').split('=')[-1]
                    total = items[-3].strip('M, ').split('=')[-1]
                    self.download_status['speed'] = speed
                    self.download_status['download'] = float(download)
                    self.download_status['total'] = float(total)
            if self.eta is None:
                if line.count('eta') > 0 and (line[-3] == ':' or
                                              line[-4] == ':'):
                    eta = items[-1].strip().split('=')[1]
                    h, m, s = [int(x) for x in eta.split(':')]
                    self.eta = h * 3600 + m * 60 + s
            if self.train_metrics is None:
                if line.count('[INFO]\t[TRAIN]') > 0 and line.count(
                        'Step') > 0:
                    if not items[-1].startswith('eta'):
                        continue
                    self.train_metrics = dict()
                    metrics = items[4:]
                    for metric in metrics:
                        try:
                            name, value = metric.strip(', ').split('=')
                            value = value.split('/')[0]
                            if value.count('.') > 0:
                                value = float(value)
                            elif value == 'nan':
                                value = 'nan'
                            else:
                                value = int(value)
                            self.train_metrics[name] = value
                        except:
                            pass
            if self.eval_metrics is None:
                if line.count('[INFO]\t[EVAL]') > 0 and line.count(
                        'Finished') > 0:
                    if not line.strip().endswith(' .'):
                        continue
                    self.eval_metrics = dict()
                    metrics = items[5:]
                    for metric in metrics:
                        try:
                            name, value = metric.strip(', ').split('=')
                            value = value.split('/')[0]
                            if value.count('.') > 0:
                                value = float(value)
                            else:
                                value = int(value)
                            self.eval_metrics[name] = value
                        except:
                            pass

        end_time_timestamp = osp.getmtime(self.log_file)
        t_diff = time.gmtime(end_time_timestamp - start_time_timestamp)
        self.running_duration = "{}小时{}分{}秒".format(
            t_diff.tm_hour, t_diff.tm_min, t_diff.tm_sec)


class PruneLogReader(object):
    def init_attr(self):
        self.eta = None
        self.iters = None
        self.current = None
        self.progress = None

    def __init__(self, log_file):
        self.log_file = log_file
        self.init_attr()

    def update(self):
        if not osp.exists(self.log_file):
            return
        logs = open(self.log_file, encoding='utf-8').read().strip().split('\n')
        self.init_attr()
        for line in logs[::-1]:
            metric_loaded = True
            for k, v in self.__dict__.items():
                if v is None:
                    metric_loaded = False
                    break
            if metric_loaded:
                break
            if line.count("Total evaluate iters") > 0:
                items = line.split(',')
                for item in items:
                    kv_list = item.strip().split()[-1].split('=')
                    kv_list = [v.strip() for v in kv_list]
                    setattr(self, kv_list[0], kv_list[1])


class QuantLogReader:
    def __init__(self, log_file):
        self.log_file = log_file
        self.stage = None
        self.running_duration = None

    def update(self):
        if not osp.exists(self.log_file):
            return
        logs = open(self.log_file, encoding='utf-8').read().strip().split('\n')
        for line in logs[::-1]:
            items = line.strip().split(' ')
            if line.count('[Run batch data]'):
                info = items[-3][:-1].split('=')[1]
                batch_id = float(info.split('/')[0])
                batch_all = float(info.split('/')[1])
                self.running_duration = \
                    batch_id / batch_all * (10.0 / 30.0)
                self.stage = 'Batch'
                break
            elif line.count('[Calculate weight]'):
                info = items[-3][:-1].split('=')[1]
                weight_id = float(info.split('/')[0])
                weight_all = float(info.split('/')[1])
                self.running_duration = \
                    weight_id / weight_all * (3.0 / 30.0) + (10.0 / 30.0)
                self.stage = 'Weight'
                break
            elif line.count('[Calculate activation]'):
                info = items[-3][:-1].split('=')[1]
                activation_id = float(info.split('/')[0])
                activation_all = float(info.split('/')[1])
                self.running_duration = \
                    activation_id / activation_all * (16.0 / 30.0) + (13.0 / 30.0)
                self.stage = 'Activation'
                break
            elif line.count('Finish quant!'):
                self.stage = 'Finish'
                break


class PyNvml(object):
    """ Nvidia GPU驱动检测类，可检测当前GPU驱动版本"""

    class PrintableStructure(Structure):
        _fmt_ = {}

        def __str__(self):
            result = []
            for x in self._fields_:
                key = x[0]
                value = getattr(self, key)
                fmt = "%s"
                if key in self._fmt_:
                    fmt = self._fmt_[key]
                elif "<default>" in self._fmt_:
                    fmt = self._fmt_["<default>"]
                result.append(("%s: " + fmt) % (key, value))
            return self.__class__.__name__ + "(" + string.join(result,
                                                               ", ") + ")"

    class c_nvmlMemory_t(PrintableStructure):
        _fields_ = [
            ('total', c_ulonglong),
            ('free', c_ulonglong),
            ('used', c_ulonglong),
        ]
        _fmt_ = {'<default>': "%d B"}

    ## Device structures
    class struct_c_nvmlDevice_t(Structure):
        pass  # opaque handle

    c_nvmlDevice_t = POINTER(struct_c_nvmlDevice_t)

    def __init__(self):
        self.nvml_lib = None
        self.nvml_lib_refcount = 0
        self.lib_load_lock = threading.Lock()
        self.nvml_lib_path = None

    def nvml_init(self, nvml_lib_path=None):
        self.lib_load_lock.acquire()
        sysstr = platform.system()
        if nvml_lib_path is None or nvml_lib_path.strip() == "":
            if sysstr == "Windows":
                nvml_lib_path = osp.join(
                    os.getenv("ProgramFiles", "C:/Program Files"),
                    "NVIDIA Corporation/NVSMI")
                if not osp.exists(osp.join(nvml_lib_path, "nvml.dll")):
                    nvml_lib_path = "C:\\Windows\\System32"

            elif sysstr == "Linux":
                p1 = "/usr/lib/x86_64-linux-gnu"
                p2 = "/usr/lib/i386-linux-gnu"
                if osp.exists(osp.join(p1, "libnvidia-ml.so.1")):
                    nvml_lib_path = p1
                elif osp.exists(osp.join(p2, "libnvidia-ml.so.1")):
                    nvml_lib_path = p2
                else:
                    nvml_lib_path = ""
            else:
                nvml_lib_path = "N/A"
        nvml_lib_dir = nvml_lib_path
        if sysstr == "Windows":
            nvml_lib_path = osp.join(nvml_lib_dir, "nvml.dll")
        else:
            nvml_lib_path = osp.join(nvml_lib_dir, "libnvidia-ml.so.1")
        self.nvml_lib_path = nvml_lib_path
        try:
            self.nvml_lib = CDLL(nvml_lib_path)
            fn = self._get_fn_ptr("nvmlInit_v2")
            fn()
            if sysstr == "Windows":
                driver_version = bytes.decode(
                    self.nvml_system_get_driver_version())
                if driver_version.strip() == "":
                    nvml_lib_path = osp.join(nvml_lib_dir, "nvml9.dll")
                    self.nvml_lib = CDLL(nvml_lib_path)
                    fn = self._get_fn_ptr("nvmlInit_v2")
                    fn()

        except Exception as e:
            raise e
        finally:
            self.lib_load_lock.release()
        self.lib_load_lock.acquire()
        self.nvml_lib_refcount += 1
        self.lib_load_lock.release()

    def create_string_buffer(self, init, size=None):
        if isinstance(init, bytes):
            if size is None:
                size = len(init) + 1
            buftype = c_char * size
            buf = buftype()
            buf.value = init
            return buf
        elif isinstance(init, int):
            buftype = c_char * init
            buf = buftype()
            return buf
        raise TypeError(init)

    def _get_fn_ptr(self, name):
        return getattr(self.nvml_lib, name)

    def nvml_system_get_driver_version(self):
        c_version = self.create_string_buffer(81)
        fn = self._get_fn_ptr("nvmlSystemGetDriverVersion")
        ret = fn(c_version, c_uint(81))
        return c_version.value

    def nvml_device_get_count(self):
        c_count = c_uint()
        fn = self._get_fn_ptr("nvmlDeviceGetCount_v2")
        ret = fn(byref(c_count))
        return c_count.value

    def nvml_device_get_handle_by_index(self, index):
        c_index = c_uint(index)
        device = PyNvml.c_nvmlDevice_t()
        fn = self._get_fn_ptr("nvmlDeviceGetHandleByIndex_v2")
        ret = fn(c_index, byref(device))
        return device

    def nvml_device_get_memory_info(self, handle):
        c_memory = PyNvml.c_nvmlMemory_t()
        fn = self._get_fn_ptr("nvmlDeviceGetMemoryInfo")
        ret = fn(handle, byref(c_memory))
        return c_memory
