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

import asyncio
import base64
import io
import mimetypes
import re
import tempfile
import uuid
from functools import partial
from typing import Awaitable, Callable, List, Optional, Tuple, TypeVar, Union, overload
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import requests
from PIL import Image
from typing_extensions import Literal, ParamSpec, TypeAlias, assert_never

from ....utils.deps import function_requires_deps, is_dep_available
from .models import ImageInfo, PDFInfo, PDFPageInfo

if is_dep_available("aiohttp"):
    import aiohttp
if is_dep_available("opencv-contrib-python"):
    import cv2
if is_dep_available("filetype"):
    import filetype
if is_dep_available("pypdfium2"):
    import pypdfium2 as pdfium
if is_dep_available("yarl"):
    import yarl

__all__ = [
    "FileType",
    "generate_log_id",
    "is_url",
    "infer_file_type",
    "infer_file_ext",
    "image_bytes_to_array",
    "image_bytes_to_image",
    "image_to_bytes",
    "image_array_to_bytes",
    "csv_bytes_to_data_frame",
    "data_frame_to_bytes",
    "base64_encode",
    "read_pdf",
    "file_to_images",
    "get_image_info",
    "write_to_temp_file",
    "get_raw_bytes",
    "get_raw_bytes_async",
    "call_async",
]

FileType: TypeAlias = Literal["IMAGE", "PDF", "VIDEO", "AUDIO"]

P = ParamSpec("P")
R = TypeVar("R")


def generate_log_id() -> str:
    return str(uuid.uuid4())


# TODO:
# 1. Use Pydantic to validate the URL and Base64-encoded string types for both
#    input and output data instead of handling this manually.
# 2. Define a `File` type for global use; this will be part of the contract.
# 3. Consider using two separate fields instead of a union of URL and Base64,
#    even though they are both strings. Backward compatibility should be
#    maintained.
def is_url(s: str) -> bool:
    if not (s.startswith("http://") or s.startswith("https://")):
        # Quick rejection
        return False
    result = urlparse(s)
    return all([result.scheme, result.netloc]) and result.scheme in ("http", "https")


def infer_file_type(url: str) -> Optional[FileType]:
    url_parts = urlparse(url)
    filename = url_parts.path.split("/")[-1]

    file_type = mimetypes.guess_type(filename)[0]

    if file_type is None:
        # HACK: The support for BOS URLs with query params is implementation-based,
        # not interface-based.
        is_bos_url = re.fullmatch(r"\w+\.bcebos\.com", url_parts.netloc) is not None
        if is_bos_url and url_parts.query:
            params = parse_qs(url_parts.query)
            if (
                "responseContentDisposition" in params
                and len(params["responseContentDisposition"]) == 1
            ):
                match_ = re.match(
                    r"attachment;filename=(.*)", params["responseContentDisposition"][0]
                )
                if match_:
                    file_type = mimetypes.guess_type(match_.group(1))[0]
        if file_type is None:
            return None

    if file_type.startswith("image/"):
        return "IMAGE"
    elif file_type == "application/pdf":
        return "PDF"
    elif file_type.startswith("video/"):
        return "VIDEO"
    elif file_type.startswith("audio/"):
        return "AUDIO"
    else:
        return None


@function_requires_deps("filetype")
def infer_file_ext(file: str) -> Optional[str]:
    if is_url(file):
        url_parts = urlparse(file)
        filename = url_parts.path.split("/")[-1]
        mime_type = mimetypes.guess_type(filename)[0]
        if mime_type is None:
            return None
        return mimetypes.guess_extension(mime_type)
    else:
        bytes_ = base64.b64decode(file)
        return "." + filetype.guess_extension(bytes_)


@function_requires_deps("opencv-contrib-python")
def image_bytes_to_array(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def image_bytes_to_image(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    with io.BytesIO() as f:
        image.save(f, format=format)
        img_bytes = f.getvalue()
    return img_bytes


@function_requires_deps("opencv-contrib-python")
def image_array_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    image = cv2.imencode(ext, image)[1]
    return image.tobytes()


def csv_bytes_to_data_frame(data: bytes) -> pd.DataFrame:
    with io.StringIO(data.decode("utf-8")) as f:
        df = pd.read_csv(f)
    return df


def data_frame_to_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


def base64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@function_requires_deps("pypdfium2", "opencv-contrib-python")
def read_pdf(
    bytes_: bytes, max_num_imgs: Optional[int] = None
) -> Tuple[List[np.ndarray], PDFInfo]:
    images: List[np.ndarray] = []
    page_info_list: List[PDFPageInfo] = []
    doc = pdfium.PdfDocument(bytes_)
    for page in doc:
        if max_num_imgs is not None and len(images) >= max_num_imgs:
            break
        # TODO: Do not always use zoom=2.0
        zoom = 2.0
        deg = 0
        image = page.render(scale=zoom, rotation=deg).to_pil()
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        images.append(image)
        page_info = PDFPageInfo(
            width=image.shape[1],
            height=image.shape[0],
        )
        page_info_list.append(page_info)
    pdf_info = PDFInfo(
        numPages=len(page_info_list),
        pages=page_info_list,
    )
    return images, pdf_info


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Tuple[List[np.ndarray], ImageInfo]: ...


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["PDF"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Tuple[List[np.ndarray], PDFInfo]: ...


@overload
def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_num_imgs: Optional[int] = ...,
) -> Union[Tuple[List[np.ndarray], ImageInfo], Tuple[List[np.ndarray], PDFInfo]]: ...


def file_to_images(
    file_bytes: bytes,
    file_type: Literal["IMAGE", "PDF"],
    *,
    max_num_imgs: Optional[int] = None,
) -> Union[Tuple[List[np.ndarray], ImageInfo], Tuple[List[np.ndarray], PDFInfo]]:
    if file_type == "IMAGE":
        images = [image_bytes_to_array(file_bytes)]
        data_info = get_image_info(images[0])
    elif file_type == "PDF":
        images, data_info = read_pdf(file_bytes, max_num_imgs=max_num_imgs)
    else:
        assert_never(file_type)
    return images, data_info


def get_image_info(image: np.ndarray) -> ImageInfo:
    return ImageInfo(width=image.shape[1], height=image.shape[0])


def write_to_temp_file(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        return f.name


def get_raw_bytes(file: str) -> bytes:
    if is_url(file):
        resp = requests.get(file, timeout=5)
        resp.raise_for_status()
        return resp.content
    else:
        return base64.b64decode(file)


@function_requires_deps("aiohttp", "yarl")
async def get_raw_bytes_async(file: str, session: "aiohttp.ClientSession") -> bytes:
    if is_url(file):
        async with session.get(yarl.URL(file, encoded=True)) as resp:
            return await resp.read()
    else:
        return base64.b64decode(file)


def call_async(
    func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> Awaitable[R]:
    return asyncio.get_running_loop().run_in_executor(
        None, partial(func, *args, **kwargs)
    )
