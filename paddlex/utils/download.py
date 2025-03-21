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
import shutil
import sys
import tarfile
import tempfile
import time
import zipfile

import requests

__all__ = ["download", "extract", "download_and_extract"]


class _ProgressPrinter(object):
    """ProgressPrinter"""

    def __init__(self, flush_interval=0.1):
        super().__init__()
        self._last_time = 0
        self._flush_intvl = flush_interval

    def print(self, str_, end=False):
        """print"""
        if end:
            str_ += "\n"
            self._last_time = 0
        if time.time() - self._last_time >= self._flush_intvl:
            sys.stdout.write(f"\r{str_}")
            self._last_time = time.time()
            sys.stdout.flush()


def _download(url, save_path, print_progress):
    if print_progress:
        print(f"Connecting to {url} ...")

    with requests.get(url, stream=True, timeout=15) as r:
        r.raise_for_status()

        total_length = r.headers.get("content-length")

        if total_length is None:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(save_path, "wb") as f:
                dl = 0
                total_length = int(total_length)
                if print_progress:
                    printer = _ProgressPrinter()
                    print(f"Downloading {os.path.basename(save_path)} ...")
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    if print_progress:
                        done = int(50 * dl / total_length)
                        printer.print(
                            f"[{'=' * done:<50s}] {float(100 * dl) / total_length:.2f}%"
                        )
            if print_progress:
                printer.print(f"[{'=' * 50:<50s}] {100:.2f}%", end=True)


def _extract_zip_file(file_path, extd_dir):
    """extract zip file"""
    with zipfile.ZipFile(file_path, "r") as f:
        file_list = f.namelist()
        total_num = len(file_list)
        for index, file in enumerate(file_list):
            f.extract(file, extd_dir)
            yield total_num, index


def _extract_tar_file(file_path, extd_dir):
    """extract tar file"""
    try:
        with tarfile.open(file_path, "r:*") as f:
            file_list = f.getnames()
            total_num = len(file_list)
            for index, file in enumerate(file_list):
                try:
                    f.extract(file, extd_dir)
                except KeyError:
                    print(f"File {file} not found in the archive.")
                yield total_num, index
    except Exception as e:
        print(f"An error occurred: {e}")


def _extract(file_path, extd_dir, print_progress):
    """extract"""
    if print_progress:
        printer = _ProgressPrinter()
        print(f"Extracting {os.path.basename(file_path)}")

    if zipfile.is_zipfile(file_path):
        handler = _extract_zip_file
    elif tarfile.is_tarfile(file_path):
        handler = _extract_tar_file
    else:
        raise RuntimeError("Unsupported file format.")

    for total_num, index in handler(file_path, extd_dir):
        if print_progress:
            done = int(50 * float(index) / total_num)
            printer.print(f"[{'=' * done:<50s}] {float(100 * index) / total_num:.2f}%")
    if print_progress:
        printer.print(f"[{'=' * 50:<50s}] {100:.2f}%", end=True)


def _remove_if_exists(path):
    """remove"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def download(url, save_path, print_progress=True, overwrite=False):
    """download"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if overwrite:
        _remove_if_exists(save_path)
    if not os.path.exists(save_path):
        _download(url, save_path, print_progress=print_progress)


def extract(file_path, extd_dir, print_progress=True):
    """extract"""
    return _extract(file_path, extd_dir, print_progress=print_progress)


def download_and_extract(
    url, save_dir, dst_name, print_progress=True, overwrite=False, no_interm_dir=True
):
    """download and extract"""
    # NOTE: `url` MUST come from a trusted source, since we do not provide a solution
    # to secure against CVE-2007-4559.
    os.makedirs(save_dir, exist_ok=True)
    dst_path = os.path.join(save_dir, dst_name)
    if overwrite:
        _remove_if_exists(dst_path)

    if not os.path.exists(dst_path):
        with tempfile.TemporaryDirectory() as td:
            arc_file_path = os.path.join(td, url.split("/")[-1])
            extd_dir = os.path.splitext(arc_file_path)[0]
            _download(url, arc_file_path, print_progress=print_progress)
            tmp_extd_dir = os.path.join(td, "extract")
            _extract(arc_file_path, tmp_extd_dir, print_progress=print_progress)
            if no_interm_dir:
                file_names = os.listdir(tmp_extd_dir)
                if len(file_names) == 1:
                    file_name = file_names[0]
                else:
                    file_name = dst_name
                sp = os.path.join(tmp_extd_dir, file_name)
                if not os.path.exists(sp):
                    raise FileNotFoundError
                dp = os.path.join(save_dir, file_name)
                if os.path.isdir(sp):
                    shutil.copytree(sp, dp, symlinks=True)
                else:
                    shutil.copyfile(sp, dp)
                extd_file = dp
            else:
                shutil.copytree(tmp_extd_dir, extd_dir)
                extd_file = extd_dir

            if not os.path.exists(dst_path) or not os.path.samefile(
                extd_file, dst_path
            ):
                shutil.move(extd_file, dst_path)
