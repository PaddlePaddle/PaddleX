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

import abc
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Discriminator, SecretStr, TypeAdapter
from typing_extensions import Annotated, Literal, assert_never

from ....utils.deps import class_requires_deps

__all__ = [
    "InMemoryStorageConfig",
    "FileSystemStorageConfig",
    "BOSConfig",
    "FileStorageConfig",
    "SupportsGetURL",
    "Storage",
    "InMemoryStorage",
    "FileSystemStorage",
    "BOS",
    "create_storage",
]


class InMemoryStorageConfig(BaseModel):
    type: Literal["memory"] = "memory"


class FileSystemStorageConfig(BaseModel):
    directory: Union[str, PathLike]

    type: Literal["file_system"] = "file_system"


class BOSConfig(BaseModel):
    endpoint: str
    ak: SecretStr
    sk: SecretStr
    bucket_name: str
    key_prefix: Optional[str] = None
    connection_timeout_in_mills: Optional[int] = None

    type: Literal["bos"] = "bos"


FileStorageConfig = Annotated[
    Union[InMemoryStorageConfig, FileSystemStorageConfig, BOSConfig],
    Discriminator("type"),
]


@runtime_checkable
class SupportsGetURL(Protocol):
    def get_url(self, key: str) -> str: ...


class Storage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get(self, key: str) -> bytes:
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, key: str, value: bytes) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        raise NotImplementedError


class InMemoryStorage(Storage):
    def __init__(self, config: InMemoryStorageConfig) -> None:
        super().__init__()
        self._data: Dict[str, bytes] = {}

    def get(self, key: str) -> bytes:
        return self._data[key]

    def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    def delete(self, key: str) -> None:
        del self._data[key]


class FileSystemStorage(Storage):
    def __init__(self, config: FileSystemStorageConfig) -> None:
        super().__init__()
        self._directory = Path(config.directory)
        self._directory.mkdir(exist_ok=True)

    def get(self, key: str) -> bytes:
        with open(self._get_file_path(key), "rb") as f:
            contents = f.read()
        return contents

    def set(self, key: str, value: bytes) -> None:
        path = self._get_file_path(key)
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            f.write(value)

    def delete(self, key: str) -> None:
        file_path = self._get_file_path(key)
        file_path.unlink(missing_ok=True)

    def _get_file_path(self, key: str) -> Path:
        return self._directory / key


@class_requires_deps("bce-python-sdk")
class BOS(Storage):
    def __init__(self, config: BOSConfig) -> None:
        from baidubce.auth.bce_credentials import BceCredentials
        from baidubce.bce_client_configuration import BceClientConfiguration
        from baidubce.services.bos.bos_client import BosClient

        super().__init__()

        bos_cfg = BceClientConfiguration(
            credentials=BceCredentials(
                config.ak.get_secret_value(), config.sk.get_secret_value()
            ),
            endpoint=config.endpoint,
            connection_timeout_in_mills=config.connection_timeout_in_mills,
        )
        self._client = BosClient(bos_cfg)
        self._bucket_name = config.bucket_name
        self._key_prefix = config.key_prefix

    def get(self, key: str) -> bytes:
        key = self._get_full_key(key)
        return self._client.get_object_as_string(bucket_name=self._bucket_name, key=key)

    def set(self, key: str, value: bytes) -> None:
        key = self._get_full_key(key)
        self._client.put_object_from_string(
            bucket=self._bucket_name, key=key, data=value
        )

    def delete(self, key: str) -> None:
        key = self._get_full_key(key)
        self._client.delete_object(bucket_name=self._bucket_name, key=key)

    def get_url(self, key: str) -> str:
        key = self._get_full_key(key)
        return self._client.generate_pre_signed_url(
            self._bucket_name, key, expiration_in_seconds=-1
        ).decode("ascii")

    def _get_full_key(self, key: str) -> str:
        if self._key_prefix:
            return f"{self._key_prefix}/{key}"
        return key


def create_storage(dic: Dict[str, Any], /) -> Storage:
    config = TypeAdapter(FileStorageConfig).validate_python(dic)
    if config.type == "memory":
        return InMemoryStorage(config)
    elif config.type == "file_system":
        return FileSystemStorage(config)
    elif config.type == "bos":
        return BOS(config)
    else:
        assert_never(config)
