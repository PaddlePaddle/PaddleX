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

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen


def _download(url, save_path):
    with urlopen(url) as r:
        with open(save_path, "wb") as file:
            shutil.copyfileobj(r, file)


def _extract_zip_file(file_path, extd_dir):
    with zipfile.ZipFile(file_path, "r") as f:
        file_list = f.namelist()
        for file in file_list:
            f.extract(file, extd_dir)


def _extract_tar_file(file_path, extd_dir):
    with tarfile.open(file_path, "r:*") as f:
        file_list = f.getnames()
        for file in file_list:
            f.extract(file, extd_dir)


def _extract(file_path, extd_dir):
    if zipfile.is_zipfile(file_path):
        handler = _extract_zip_file
    elif tarfile.is_tarfile(file_path):
        handler = _extract_tar_file
    else:
        raise ValueError("Unsupported file format")
    handler(file_path, extd_dir)


def _remove_if_exists(path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def download(url, save_path, overwrite=False):
    save_path.parent.mkdir(exist_ok=True)
    if overwrite:
        _remove_if_exists(save_path)
    if not save_path.exists():
        _download(url, save_path)


def extract(file_path, extd_dir):
    return _extract(file_path, extd_dir)


def download_and_extract(url, save_dir, dst_name, overwrite=False, no_interm_dir=True):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    dst_path = save_dir / dst_name
    if overwrite:
        _remove_if_exists(dst_path)

    if not dst_path.exists():
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            arc_file_path = td / url.split("/")[-1]
            extd_dir = arc_file_path.stem
            _download(url, arc_file_path)
            tmp_extd_dir = td / "extracted"
            _extract(arc_file_path, tmp_extd_dir)
            if no_interm_dir:
                paths = list(tmp_extd_dir.iterdir())
                if len(paths) == 1:
                    sp = paths[0]
                else:
                    sp = tmp_extd_dir / dst_name
                if not sp.exists():
                    raise FileNotFoundError
                dp = save_dir / sp.name
                if sp.is_dir():
                    shutil.copytree(sp, dp)
                else:
                    shutil.copyfile(sp, dp)
                extd_file = dp
            else:
                shutil.copytree(tmp_extd_dir, extd_dir)
                extd_file = extd_dir

            if not dst_path.exists() or not extd_file.samefile(dst_path):
                shutil.move(extd_file, dst_path)
