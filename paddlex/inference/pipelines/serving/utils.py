# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import asyncio
import base64
import io
import uuid
from urllib.parse import urlparse
from typing import Callable, Awaitable, List, Optional, TypeVar
from functools import partial

import aiohttp
import fitz
import cv2
import numpy as np
import pandas as pd
import yarl
from PIL import Image
from typing_extensions import ParamSpec


_P = ParamSpec("_P")
_R = TypeVar("_R")


def generate_log_id() -> str:
    return str(uuid.uuid4())


def is_url(s: str) -> bool:
    if not (s.startswith("http://") or s.startswith("https://")):
        # Quick rejection
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


async def get_raw_bytes(file: str, session: aiohttp.ClientSession) -> bytes:
    if is_url(file):
        async with session.get(yarl.URL(file, encoded=True)) as resp:
            return await resp.read()
    else:
        return base64.b64decode(file)


def image_bytes_to_array(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def image_to_base64(image: Image.Image) -> str:
    with io.BytesIO() as f:
        image.save(f, format="JPEG")
        image_base64 = base64.b64encode(f.getvalue()).decode("ascii")
    return image_base64


def csv_bytes_to_data_frame(data: bytes) -> pd.DataFrame:
    with io.StringIO(data.decode("utf-8")) as f:
        df = pd.read_csv(f)
    return df


def data_frame_to_base64(df: str) -> str:
    return base64.b64encode(df.to_csv()).decode("ascii")


def read_pdf(
    bytes_: bytes, resize: bool = False, max_num_imgs: Optional[int] = None
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    img_size = None
    with fitz.open("pdf", bytes_) as doc:
        for page in doc:
            if max_num_imgs is not None and len(images) >= max_num_imgs:
                break
            # TODO: Do not always use zoom=2.0
            zoom = 2.0
            deg = 0
            mat = fitz.Matrix(zoom, zoom).prerotate(deg)
            pixmap = page.get_pixmap(matrix=mat, alpha=False)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.h, pixmap.w, pixmap.n
            )
            image = np.ascontiguousarray(image[..., ::-1])
            if resize:
                if img_size is None:
                    img_size = (image.shape[1], image.shape[0])
                else:
                    if (image.shape[1], image.shape[0]) != img_size:
                        image = cv2.resize(image, img_size)
            images.append(image)
    return images


def async_call(
    func: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> Awaitable[_R]:
    return asyncio.get_running_loop().run_in_executor(
        None, partial(func, *args, **kwargs)
    )
