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

import copy
import json
import mimetypes
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from ....utils import logging
from ...utils.io import (
    CSVWriter,
    HtmlWriter,
    ImageWriter,
    JsonWriter,
    MarkdownWriter,
    TextWriter,
    VideoWriter,
    XlsxWriter,
)


class StrMixin:
    """Mixin class for adding string conversion capabilities."""

    @property
    def str(self) -> Dict[str, str]:
        """Property to get the string representation of the result.

        Returns:
            Dict[str, str]: The string representation of the result.
        """

        return self._to_str()

    def _to_str(
        self,
    ):
        """Convert the given result data to a string representation.

        Args:
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.

        Returns:
            Dict[str, str]: The string representation of the result.
        """
        return {"res": self}

    def print(self) -> None:
        """Print the string representation of the result."""
        logging.info(self._to_str())


def _format_data(obj):
    """Helper function to format data into a JSON-serializable format.

    Args:
        obj: The object to be formatted.

    Returns:
        Any: The formatted object.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_format_data(item) for item in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", force_ascii=False))
    elif isinstance(obj, Path):
        return obj.as_posix()
    elif isinstance(obj, dict):
        return dict({k: _format_data(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return [_format_data(i) for i in obj]
    else:
        return obj


class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        self._save_funcs.append(self.save_to_json)

    def _to_json(self) -> Dict[str, Dict[str, Any]]:
        """Convert the object to a JSON-serializable format.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary representation of the object that is JSON-serializable.
        """

        return {"res": _format_data(copy.deepcopy(self))}

    @property
    def json(self) -> Dict[str, Dict[str, Any]]:
        """Property to get the JSON representation of the result.

        Returns:
            Dict[str, Dict[str, Any]]: The dict type JSON representation of the result.
        """

        return self._to_json()

    def save_to_json(
        self,
        save_path: str,
        indent: int = 4,
        ensure_ascii: bool = False,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Save the JSON representation of the object to a file.

        Args:
            save_path (str): The path to save the JSON file. If the save path does not end with '.json', it appends the base name and suffix of the input path.
            indent (int): The number of spaces to indent for pretty printing. Default is 4.
            ensure_ascii (bool): If False, non-ASCII characters will be included in the output. Default is False.
            *args: Additional positional arguments to pass to the underlying writer.
            **kwargs: Additional keyword arguments to pass to the underlying writer.
        """

        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"

        json_data = self._to_json()
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                save_path = base_save_path / f"{stem}_{key}.json"
                self._json_writer.write(
                    save_path.as_posix(),
                    json_data[key],
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    *args,
                    **kwargs,
                )
        else:
            if len(json_data) > 1:
                logging.warning(
                    f"The result has multiple json files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._json_writer.write(
                save_path,
                json_data[list(json_data.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
                *args,
                **kwargs,
            )

    def _to_str(
        self,
        json_format: bool = False,
        indent: int = 4,
        ensure_ascii: bool = False,
    ):
        """Convert the given result data to a string representation.
        Args:
            data (dict): The data would be converted to str.
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        Returns:
            Dict[str, str]: The string representation of the result.
        """
        if json_format:
            return json.dumps(
                _format_data({"res": self}), indent=indent, ensure_ascii=ensure_ascii
            )
        else:
            return {"res": self}

    def print(
        self, json_format: bool = False, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Print the string representation of the result.

        Args:
            json_format (bool): If True, print a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        """
        str_ = self._to_str(
            json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logging.info(str_)


class Base64Mixin:
    """Mixin class for adding Base64 encoding capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the Base64Mixin.

        Args:
            *args: Positional arguments to pass to the TextWriter.
            **kwargs: Keyword arguments to pass to the TextWriter.
        """
        self._base64_writer = TextWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_base64)

    @abstractmethod
    def _to_base64(self) -> Dict[str, str]:
        """Abstract method to convert the result to Base64.

        Returns:
            Dict[str, str]: The str type Base64 representation result.
        """
        raise NotImplementedError

    @property
    def base64(self) -> Dict[str, str]:
        """
        Property that returns the Base64 encoded content.

        Returns:
            Dict[str, str]: The base64 representation of the result.
        """
        return self._to_base64()

    def save_to_base64(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the Base64 encoded content to the specified path.

        Args:
            save_path (str): The path to save the base64 representation result. If the save path does not end with '.b64', it appends the base name and suffix of the input path.

            *args: Additional positional arguments that will be passed to the base64 writer.
            **kwargs: Additional keyword arguments that will be passed to the base64 writer.
        """
        base64 = self._to_base64()
        if not str(save_path).lower().endswith((".b64")):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in base64:
                save_path = base_save_path / f"{stem}_{key}.b64"
                self._base64_writer.write(
                    save_path.as_posix(), base64[key], *args, **kwargs
                )
        else:
            if len(base64) > 1:
                logging.warning(
                    f"The result has multiple base64 files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._base64_writer.write(
                save_path, base64[list(base64.keys())[0]], *args, **kwargs
            )


class ImgMixin:
    """Mixin class for adding image handling capabilities."""

    def __init__(self, backend: str = "pillow", *args: List, **kwargs: Dict) -> None:
        """Initializes ImgMixin.

        Args:
            backend (str): The backend to use for image processing. Defaults to "pillow".
            *args: Additional positional arguments to pass to the ImageWriter.
            **kwargs: Additional keyword arguments to pass to the ImageWriter.
        """
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._save_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self) -> Dict[str, Image.Image]:
        """Abstract method to convert the result to an image.

        Returns:
            Dict[str, Image.Image]: The image representation result.
        """
        raise NotImplementedError

    @property
    def img(self) -> Dict[str, Image.Image]:
        """Property to get the image representation of the result.

        Returns:
            Dict[str, Image.Image]: The image representation of the result.
        """
        return self._to_img()

    def save_to_img(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the image representation of the result to the specified path.

        Args:
            save_path (str): The path to save the image. If the save path does not end with .jpg or .png, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the image writer.
            **kwargs: Additional keyword arguments that will be passed to the image writer.
        """

        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")

        img = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                self._img_writer.write(save_path.as_posix(), img[key], *args, **kwargs)
        else:
            if len(img) > 1:
                logging.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._img_writer.write(save_path, img[list(img.keys())[0]], *args, **kwargs)


class CSVMixin:
    """Mixin class for adding CSV handling capabilities."""

    def __init__(self, backend: str = "pandas", *args: List, **kwargs: Dict) -> None:
        """Initializes the CSVMixin.

        Args:
            backend (str): The backend to use for CSV operations (default is "pandas").
            *args: Optional positional arguments to pass to the CSVWriter.
            **kwargs: Optional keyword arguments to pass to the CSVWriter.
        """
        self._csv_writer = CSVWriter(backend=backend, *args, **kwargs)
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        self._save_funcs.append(self.save_to_csv)

    @property
    def csv(self) -> Dict[str, pd.DataFrame]:
        """Property to get the pandas Dataframe representation of the result.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation of the result.
        """
        return self._to_csv()

    @abstractmethod
    def _to_csv(self) -> Dict[str, pd.DataFrame]:
        """Abstract method to convert the result to pandas.DataFrame.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation result.
        """
        raise NotImplementedError

    def save_to_csv(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the result to a CSV file.

        Args:
            save_path (str): The path to save the CSV file. If the path does not end with ".csv",
                the stem of the input path attribute (self['input_path']) will be used as the filename.
            *args: Optional positional arguments to pass to the CSV writer's write method.
            **kwargs: Optional keyword arguments to pass to the CSV writer's write method.
        """

        def _is_csv_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/csv"

        csv = self._to_csv()
        if not _is_csv_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in csv:
                save_path = base_save_path / f"{stem}_{key}.csv"
                self._csv_writer.write(save_path.as_posix(), csv[key], *args, **kwargs)
        else:
            if len(csv) > 1:
                logging.warning(
                    f"The result has multiple csv files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._csv_writer.write(save_path, csv[list(csv.keys())[0]], *args, **kwargs)


class HtmlMixin:
    """Mixin class for adding HTML handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """
        Initializes the HTML writer and appends the save_to_html method to the save functions list.

        Args:
            *args: Positional arguments passed to the HtmlWriter.
            **kwargs: Keyword arguments passed to the HtmlWriter.
        """
        self._html_writer = HtmlWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_html)

    @property
    def html(self) -> Dict[str, str]:
        """Property to get the HTML representation of the result.

        Returns:
            str: The str type HTML representation of the result.
        """
        return self._to_html()

    @abstractmethod
    def _to_html(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_html(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation of the object to the specified path.

        Args:
            save_path (str): The path to save the HTML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        def _is_html_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/html"

        html = self._to_html()
        if not _is_html_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in html:
                save_path = base_save_path / f"{stem}_{key}.html"
                self._html_writer.write(
                    save_path.as_posix(), html[key], *args, **kwargs
                )
        else:
            if len(html) > 1:
                logging.warning(
                    f"The result has multiple html files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._html_writer.write(
                save_path, html[list(html.keys())[0]], *args, **kwargs
            )


class XlsxMixin:
    """Mixin class for adding XLSX handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the XLSX writer and appends the save_to_xlsx method to the save functions.

        Args:
            *args: Positional arguments to be passed to the XlsxWriter constructor.
            **kwargs: Keyword arguments to be passed to the XlsxWriter constructor.
        """
        self._xlsx_writer = XlsxWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_xlsx)

    @property
    def xlsx(self) -> Dict[str, str]:
        """Property to get the XLSX representation of the result.

        Returns:
            Dict[str, str]: The str type XLSX representation of the result.
        """
        return self._to_xlsx()

    @abstractmethod
    def _to_xlsx(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type XLSX representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_xlsx(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation to an XLSX file.

        Args:
            save_path (str): The path to save the XLSX file. If the path does not end with ".xlsx",
                             the filename will be set to the stem of the input path with ".xlsx" extension.
            *args: Additional positional arguments to pass to the XLSX writer.
            **kwargs: Additional keyword arguments to pass to the XLSX writer.
        """

        def _is_xlsx_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return (
                mime_type is not None
                and mime_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        xlsx = self._to_xlsx()
        if not _is_xlsx_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in xlsx:
                save_path = base_save_path / f"{stem}_{key}.xlsx"
                self._xlsx_writer.write(
                    save_path.as_posix(), xlsx[key], *args, **kwargs
                )
        else:
            if len(xlsx) > 1:
                logging.warning(
                    f"The result has multiple xlsx files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._xlsx_writer.write(
                save_path, xlsx[list(xlsx.keys())[0]], *args, **kwargs
            )


class VideoMixin:
    """Mixin class for adding Video handling capabilities."""

    def __init__(self, backend: str = "opencv", *args: List, **kwargs: Dict) -> None:
        """Initializes VideoMixin.

        Args:
            backend (str): The backend to use for video processing. Defaults to "opencv".
            *args: Additional positional arguments to pass to the VideoWriter.
            **kwargs: Additional keyword arguments to pass to the VideoWriter.
        """
        self._backend = backend
        self._save_funcs.append(self.save_to_video)

    @abstractmethod
    def _to_video(self) -> Dict[str, np.array]:
        """Abstract method to convert the result to a video.

        Returns:
            Dict[str, np.array]: The video representation result.
        """
        raise NotImplementedError

    @property
    def video(self) -> Dict[str, np.array]:
        """Property to get the video representation of the result.

        Returns:
            Dict[str, np.array]: The video representation of the result.
        """
        return self._to_video()

    def save_to_video(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the video representation of the result to the specified path.

        Args:
            save_path (str): The path to save the video. If the save path does not end with .mp4 or .avi, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the video writer.
            **kwargs: Additional keyword arguments that will be passed to the video writer.
        """

        def _is_video_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("video/")

        video_writer = VideoWriter(backend=self._backend, *args, **kwargs)
        video = self._to_video()
        if not _is_video_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            suffix = fn.suffix if _is_video_file(fn) else ".mp4"
            base_save_path = Path(save_path)
            for key in video:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                video_writer.write(save_path.as_posix(), video[key], *args, **kwargs)
        else:
            if len(video) > 1:
                logging.warning(
                    f"The result has multiple video files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            video_writer.write(save_path, video[list(video.keys())[0]], *args, **kwargs)


class MarkdownMixin:
    """Mixin class for adding Markdown handling capabilities."""

    def __init__(self, *args: list, **kwargs: dict):
        """Initializes the Markdown writer and appends the save_to_markdown method to the save functions.

        Args:
            *args: Positional arguments to be passed to the MarkdownWriter constructor.
            **kwargs: Keyword arguments to be passed to the MarkdownWriter constructor.
        """
        self._markdown_writer = MarkdownWriter(*args, **kwargs)
        self._img_writer = ImageWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_markdown)

    @abstractmethod
    def _to_markdown(self, pretty=True) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Convert the result to markdown format.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.

        Returns:
            Dict[str, Union[str, Dict[str, Any]]]: A dictionary containing markdown text and image data.
        """
        raise NotImplementedError

    @property
    def markdown(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Property to access the markdown data.

        Returns:
            Dict[str, Union[str, Dict[str, Any]]]: A dictionary containing markdown text and image data.
        """
        return self._to_markdown()

    def save_to_markdown(self, save_path, pretty=True, *args, **kwargs) -> None:
        """Save the markdown data to a file.

        Args:
            save_path (Union[str, Path]): The path where the markdown file will be saved.
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """

        def _is_markdown_file(file_path) -> bool:
            """Check if a file is a markdown file based on its extension or MIME type.

            Args:
                file_path (Union[str, Path]): The path to the file.

            Returns:
                bool: True if the file is a markdown file, False otherwise.
            """
            markdown_extensions = {".md", ".markdown", ".mdown", ".mkd"}
            _, ext = os.path.splitext(str(file_path))
            if ext.lower() in markdown_extensions:
                return True
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type == "text/markdown"

        if not _is_markdown_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_markdown_file(fn) else ".md"
            stem = fn.stem
            base_save_path = Path(save_path)
            save_path = base_save_path / f"{stem}{suffix}"
            self.save_path = save_path
        else:
            self.save_path = save_path
        self._save_data(
            self._markdown_writer.write,
            self._img_writer.write,
            self.save_path,
            self._to_markdown(pretty=pretty),
            *args,
            **kwargs,
        )

    def _save_data(
        self,
        save_mkd_func: Callable,
        save_img_func: Callable,
        save_path: Union[str, Path],
        data: Optional[Dict[str, Union[str, Dict[str, Any]]]],
        *args,
        **kwargs,
    ) -> None:
        """Internal method to save markdown and image data.

        Args:
            save_mkd_func (Callable): Function to save markdown text.
            save_img_func (Callable): Function to save image data.
            save_path (Union[str, Path]): The base path where the data will be saved.
            data (Optional[Dict[str, Union[str, Dict[str, Any]]]]): The markdown data to save.
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """
        save_path = Path(save_path)
        if data is None:
            return
        for key, value in data.items():
            if isinstance(value, str):
                save_mkd_func(save_path.as_posix(), value, *args, **kwargs)
            if isinstance(value, dict):
                base_save_path = save_path.parent
                for img_path, img_data in value.items():
                    save_img_func(
                        (base_save_path / img_path).as_posix(),
                        img_data,
                        *args,
                        **kwargs,
                    )
