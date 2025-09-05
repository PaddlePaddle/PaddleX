# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import base64
import mimetypes
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen


def is_url(s):
    if not (s.startswith("http://") or s.startswith("https://")):
        # Quick rejection
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def prepare_input_file(file, include_header=False):
    if is_url(file):
        return file
    else:
        with open(file, "rb") as f:
            bytes_ = f.read()
        encoded = base64.b64encode(bytes_).decode("ascii")
        if include_header:
            mime_type = mimetypes.guess_type(file)[0] or "application/octet-stream"
            return f"data:{mime_type};base64,{encoded}"
        return encoded


def save_output_file(file, path, include_header=False):
    if is_url(file):
        with urlopen(file) as r:
            with open(path, "wb") as f:
                shutil.copyfileobj(r, f)
    else:
        if include_header:
            header, encoded = file.split(",", 1)
            if not (header.startswith("data:") and header.endswith(";base64")):
                raise ValueError("Invalid data URI format")
        else:
            encoded = file
        bytes_ = base64.b64decode(encoded)
        with open(path, "wb") as f:
            f.write(bytes_)
