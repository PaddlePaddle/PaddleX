# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import time
import zipfile
import tarfile
import hashlib
import tqdm
import logging

from .utils.hub_model_server import model_server
from .utils import hub_env as hubenv

DOWNLOAD_RETRY_LIMIT = 3


def md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logging.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logging.info(
            "File {} md5 check failed, {}(calc) != "
            "{}(base)".format(fullname, calc_md5sum, md5sum)
        )
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
        if not osp.isdir(src):
            shutil.move(src, dst)
            return
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)


def download(url, path, rename=None, md5sum=None, show_progress=False):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    if rename is not None:
        fullname = osp.join(path, rename)
    retry_cnt = 0
    while not (osp.exists(fullname) and md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            logging.debug("{} download failed.".format(fname))
            raise RuntimeError(
                "Download from {} failed. " "Retry limit reached".format(url)
            )

        logging.info("Downloading {} from {}".format(fname, url))

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError(
                "Downloading from {} failed with code "
                "{}!".format(url, req.status_code)
            )

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get("content-length")
        with open(tmp_fullname, "wb") as f:
            if total_size and show_progress:
                for chunk in tqdm.tqdm(
                    req.iter_content(chunk_size=1024),
                    total=(int(total_size) + 1023) // 1024,
                    unit="KB",
                ):
                    f.write(chunk)
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
    fpath_tmp = osp.join(fpath, "tmp")
    if osp.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find(".tar") >= 0 or fname.find(".tgz") >= 0:
        with tarfile.open(fname) as tf:

            def is_within_directory(directory, target):

                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)

                prefix = os.path.commonprefix([abs_directory, abs_target])

                return prefix == abs_directory

            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")

                tar.extractall(path, members, numeric_owner=numeric_owner)

            safe_extract(tf, path=fpath_tmp)
    elif fname.find(".zip") >= 0:
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


def url2dir(url, path, rename=None):
    full_name = download(url, path, rename, show_progress=True)
    print("File is donwloaded, now extracting...")
    if url.count(".tgz") > 0 or url.count(".tar") > 0 or url.count("zip") > 0:
        return decompress(full_name)


def download_and_decompress(url, path=".", rename=None):
    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    # if url.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
    #     fullname = osp.join(path, fname.split('.')[0])
    nranks = 0
    if nranks <= 1:
        dst_dir = url2dir(url, path, rename)
        if dst_dir is not None:
            fullname = dst_dir
    else:
        lock_path = fullname + ".lock"
        if not os.path.exists(fullname):
            with open(lock_path, "w"):
                os.utime(lock_path, None)
            if nranks == 0:
                dst_dir = url2dir(url, path, rename)
                if dst_dir is not None:
                    fullname = dst_dir
                os.remove(lock_path)
            else:
                while os.path.exists(lock_path):
                    time.sleep(1)
    return


def get_model_list(category: str = None):
    """
    Get all pre-trained models information supported by fd.download_model.
    Args:
        category(str): model category, if None, list all models in all categories.
    Returns:
        results(dict): a dictionary, key is category, value is a list which contains models information.
    """
    result = model_server.get_model_list()
    if result["status"] != 0:
        raise ValueError(
            "Failed to get pretrained models information from hub model server."
        )
    result = result["data"]
    if category is None:
        return result
    elif category in result:
        return {category: result[category]}
    else:
        raise ValueError(
            "No pretrained model in category {} can be downloaded now.".format(category)
        )


def download_model(
    name: str, path: str = None, format: str = None, version: str = None
):
    """
    Download pre-trained model for UltraInfer inference engine.
    Args:
        name: model name
        path(str): local path for saving model. If not set, default is hubenv.MODEL_HOME
        format(str): UltraInfer model format
        version(str) : UltraInfer model version
    """
    result = model_server.search_model(name, format, version)
    if path is None:
        path = hubenv.MODEL_HOME
    if result:
        url = result[0]["url"]
        format = result[0]["format"]
        version = result[0]["version"]
        fullpath = download(url, path, show_progress=True)
        model_server.stat_model(name, format, version)
        if format == "paddle":
            if url.count(".tgz") > 0 or url.count(".tar") > 0 or url.count("zip") > 0:
                archive_path = fullpath
                fullpath = decompress(fullpath)
                try:
                    os.rename(fullpath, os.path.join(os.path.dirname(fullpath), name))
                    fullpath = os.path.join(os.path.dirname(fullpath), name)
                    os.remove(archive_path)
                except FileExistsError:
                    pass
        print("Successfully download model at path: {}".format(fullpath))
        return fullpath
    else:
        print("ERROR: Could not find a model named {}".format(name))
