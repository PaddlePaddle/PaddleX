#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import requests
import shutil
import tarfile
import tqdm
import zipfile

from paddlex.ppcls.modeling import similar_architectures
from paddlex.ppcls.utils import logger

__all__ = ['get']

DOWNLOAD_RETRY_LIMIT = 3


class UrlError(Exception):
    """ UrlError
    """

    def __init__(self, url='', code=''):
        message = "Downloading from {} failed with code {}!".format(url, code)
        super(UrlError, self).__init__(message)


class ModelNameError(Exception):
    """ ModelNameError
    """

    def __init__(self, message=''):
        super(ModelNameError, self).__init__(message)


class RetryError(Exception):
    """ RetryError
    """

    def __init__(self, url='', times=''):
        message = "Download from {} failed. Retry({}) limit reached".format(
            url, times)
        super(RetryError, self).__init__(message)


def _get_url(architecture, postfix="pdparams"):
    prefix = "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/"
    fname = architecture + "_pretrained." + postfix
    return prefix + fname


def _move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not os.path.exists(dst):
        shutil.move(src, dst)
    elif os.path.isfile(src):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = os.path.join(src, fp)
            dst_fp = os.path.join(dst, fp)
            if os.path.isdir(src_fp):
                if os.path.isdir(dst_fp):
                    _move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif os.path.isfile(src_fp) and \
                    not os.path.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)


def _download(url, path):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    if not os.path.exists(path):
        os.makedirs(path)

    fname = os.path.split(url)[-1]
    fullname = os.path.join(path, fname)
    retry_cnt = 0

    while not os.path.exists(fullname):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RetryError(url, DOWNLOAD_RETRY_LIMIT)

        logger.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise UrlError(url, req.status_code)

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

    return fullname


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.
    fpath = os.path.split(fname)[0]
    fpath_tmp = os.path.join(fpath, 'tmp')
    if os.path.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('tar') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    fs = os.listdir(fpath_tmp)
    assert len(
        fs
    ) == 1, "There should just be 1 pretrained path in an archive file but got {}.".format(
        len(fs))

    f = fs[0]
    src_dir = os.path.join(fpath_tmp, f)
    dst_dir = os.path.join(fpath, f)
    _move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    os.remove(fname)

    return f


def _get_pretrained():
    with open('./ppcls/utils/pretrained.list') as flist:
        pretrained = [line.strip() for line in flist]
    return pretrained


def _check_pretrained_name(architecture):
    assert isinstance(architecture, str), \
        ("the type of architecture({}) should be str". format(architecture))
    pretrained = _get_pretrained()
    similar_names = similar_architectures(architecture, pretrained)
    model_list = ', '.join(similar_names)
    err = "{} is not exist! Maybe you want: [{}]" \
          "".format(architecture, model_list)
    if architecture not in similar_names:
        raise ModelNameError(err)


def list_models():
    pretrained = _get_pretrained()
    msg = "All avialable pretrained models are as follows: {}".format(
        pretrained)
    logger.info(msg)
    return


def get(architecture, path, decompress=False, postfix="pdparams"):
    """
    Get the pretrained model.
    """
    _check_pretrained_name(architecture)
    url = _get_url(architecture, postfix=postfix)
    fname = _download(url, path)
    if postfix == "tar" and decompress:
        _decompress(fname)
    logger.info("download {} finished ".format(fname))
