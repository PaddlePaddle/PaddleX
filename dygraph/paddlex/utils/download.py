# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp
import shutil
import requests
import tqdm
import time
import hashlib
import tarfile
import zipfile
import filelock
import paddle
from . import logging

DOWNLOAD_RETRY_LIMIT = 3


def md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logging.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logging.info("File {} md5 check failed, {}(calc) != "
                     "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not osp.exists(dst):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and \
                    not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)


def download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0
    while not (osp.exists(fullname) and md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            logging.debug("{} download failed.".format(fname))
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        logging.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                download_size = 0
                current_time = time.time()
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
                    download_size += 1024
                    if download_size % 524288 == 0:
                        total_size_m = round(
                            int(total_size) / 1024.0 / 1024.0, 2)
                        download_size_m = round(download_size / 1024.0 /
                                                1024.0, 2)
                        speed = int(524288 /
                                    (time.time() - current_time + 0.01) /
                                    1024.0)
                        current_time = time.time()
                        logging.debug(
                            "Downloading: TotalSize={}M, DownloadSize={}M, Speed={}KB/s"
                            .format(total_size_m, download_size_m, speed))
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
        logging.debug("{} download completed.".format(fname))

    return fullname


def decompress(fname):
    """
    Decompress for zip and tar file
    """
    logging.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.
    fpath = osp.split(fname)[0]
    fpath_tmp = osp.join(fpath, 'tmp')
    if osp.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('tar') >= 0 or fname.find('tgz') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    for f in os.listdir(fpath_tmp):
        src_dir = osp.join(fpath_tmp, f)
        dst_dir = osp.join(fpath, f)
        move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    logging.debug("{} decompressed.".format(fname))
    return dst_dir


def url2dir(url, path):
    download(url, path)
    if url.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        fname = osp.split(url)[-1]
        savepath = osp.join(path, fname)
        return decompress(savepath)


def download_and_decompress(url, path='.'):
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    # if url.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
    #     fullname = osp.join(path, fname.split('.')[0])
    if nranks <= 1:
        dst_dir = url2dir(url, path)
        if dst_dir is not None:
            fullname = dst_dir
    else:
        lock_path = fullname + '.lock'
        if not os.path.exists(fullname):
            with open(lock_path, 'w'):
                os.utime(lock_path, None)
            if local_rank == 0:
                dst_dir = url2dir(url, path)
                if dst_dir is not None:
                    fullname = dst_dir
                os.remove(lock_path)
            else:
                while os.path.exists(lock_path):
                    time.sleep(1)
    return fullname
