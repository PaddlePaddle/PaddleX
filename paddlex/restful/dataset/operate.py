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

import os
import pickle
import traceback
import os.path as osp
import multiprocessing as mp
from .cls_dataset import ClsDataset
from .det_dataset import DetDataset
from .seg_dataset import SegDataset
from .ins_seg_dataset import InsSegDataset
from ..utils import set_folder_status, get_folder_status, DatasetStatus, DownloadStatus, download, list_files

dataset_url_list = [
    'https://bj.bcebos.com/paddlex/demos/vegetables_cls.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/insect_det.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/optic_disc_seg.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/xiaoduxiong_ins_det.tar.gz',
    'https://bj.bcebos.com/paddlex/demos/remote_sensing_seg.tar.gz'
]
dataset_url_dict = {
    'classification':
    'https://bj.bcebos.com/paddlex/demos/vegetables_cls.tar.gz',
    'detection': 'https://bj.bcebos.com/paddlex/demos/insect_det.tar.gz',
    'segmentation':
    'https://bj.bcebos.com/paddlex/demos/optic_disc_seg.tar.gz',
    'instance_segmentation':
    'https://bj.bcebos.com/paddlex/demos/xiaoduxiong_ins_det.tar.gz'
}


def _check_and_copy(dataset, dataset_path, source_path):
    try:
        dataset.check_dataset(source_path)
    except Exception as e:
        error_info = traceback.format_exc()
        set_folder_status(dataset_path, DatasetStatus.XCHECKFAIL, error_info)
        return
    set_folder_status(dataset_path, DatasetStatus.XCOPYING, os.getpid())
    try:
        dataset.copy_dataset(source_path, dataset.all_files)
    except Exception as e:
        error_info = traceback.format_exc()
        set_folder_status(dataset_path, DatasetStatus.XCOPYFAIL, error_info)
        return
    # 若上传已切分好的数据集
    if len(dataset.train_files) != 0:
        set_folder_status(dataset_path, DatasetStatus.XSPLITED)


def import_dataset(dataset_id, dataset_type, dataset_path, source_path):
    set_folder_status(dataset_path, DatasetStatus.XCHECKING)
    if dataset_type == 'classification':
        ds = ClsDataset(dataset_id, dataset_path)
    elif dataset_type == 'detection':
        ds = DetDataset(dataset_id, dataset_path)
    elif dataset_type == 'segmentation':
        ds = SegDataset(dataset_id, dataset_path)
    elif dataset_type == 'instance_segmentation':
        ds = InsSegDataset(dataset_id, dataset_path)
    p = mp.Process(
        target=_check_and_copy, args=(ds, dataset_path, source_path))
    p.start()
    return p


def _download_proc(url, target_path, dataset_type):
    # 下载数据集压缩包
    from paddlex.utils import decompress
    target_path = osp.join(target_path, dataset_type)
    fname = download(url, target_path)
    # 解压
    decompress(fname)
    set_folder_status(target_path, DownloadStatus.XDDECOMPRESSED)


def download_demo_dataset(prj_type, target_path):
    url = dataset_url_list[prj_type.value]
    dataset_type = prj_type.name
    p = mp.Process(
        target=_download_proc, args=(url, target_path, dataset_type))
    p.start()
    return p


def get_dataset_status(dataset_id, dataset_type, dataset_path):
    status, message = get_folder_status(dataset_path, True)
    if status is None:
        status = DatasetStatus.XEMPTY
    if status == DatasetStatus.XCOPYING:
        items = message.strip().split()
        pid = None
        if len(items) < 2:
            percent = 0.0
        else:
            pid = int(items[0])
            if int(items[1]) == 0:
                percent = 1.0
            else:
                copyed_files_num = len(list_files(dataset_path)) - 1
                percent = copyed_files_num * 1.0 / int(items[1])
        message = {'pid': pid, 'percent': percent}
    if status == DatasetStatus.XCOPYDONE or status == DatasetStatus.XSPLITED:
        if not osp.exists(osp.join(dataset_path, 'statis.pkl')):
            p = import_dataset(dataset_id, dataset_type, dataset_path,
                               dataset_path)
            status = DatasetStatus.XCHECKING
    return status, message


def split_dataset(dataset_id, dataset_type, dataset_path, val_split,
                  test_split):
    status, message = get_folder_status(dataset_path, True)
    if status != DatasetStatus.XCOPYDONE and status != DatasetStatus.XSPLITED:
        raise Exception("数据集还未导入完成，请等数据集导入成功后再进行切分")
    if not osp.exists(osp.join(dataset_path, 'statis.pkl')):
        raise Exception("数据集需重新校验，请刷新数据集后再进行切分")

    if dataset_type == 'classification':
        ds = ClsDataset(dataset_id, dataset_path)
    elif dataset_type == 'detection':
        ds = DetDataset(dataset_id, dataset_path)
    elif dataset_type == 'segmentation':
        ds = SegDataset(dataset_id, dataset_path)
    elif dataset_type == 'instance_segmentation':
        ds = InsSegDataset(dataset_id, dataset_path)

    ds.load_statis_info()
    ds.split(val_split, test_split)
    set_folder_status(dataset_path, DatasetStatus.XSPLITED)


def get_dataset_details(dataset_path):
    status, message = get_folder_status(dataset_path, True)
    if status == DatasetStatus.XCOPYDONE or status == DatasetStatus.XSPLITED:
        with open(osp.join(dataset_path, 'statis.pkl'), 'rb') as f:
            details = pickle.load(f)
        return details
    return None
