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

import base64
import uuid
from typing import Any, Dict, Union, Optional, Literal

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient
from pydantic import TypeAdapter, Discriminator
from typing_extensions import Annotated, assert_never
from pydantic import BaseModel, SecretStr


class InMemoryStorageConfig(BaseModel):
    type: Literal["memory"] = "memory"


class BOSConfig(BaseModel):
    endpoint: str
    ak: SecretStr
    sk: SecretStr
    bucket_name: str
    key_prefix: Optional[str] = None
    connection_timeout_in_mills: Optional[int] = None

    type: Literal["bos"] = "bos"


FileStorageConfig = Union[InMemoryStorageConfig, BOSConfig]


def parse_file_storage_config(dic: Dict[str, Any]) -> FileStorageConfig:
    # XXX: mypy deduces a wrong type
    return TypeAdapter(
        Annotated[FileStorageConfig, Discriminator("type")]
    ).validate_python(
        dic
    )  # type: ignore[return-value]


def postprocess_file(
    file: bytes, config: FileStorageConfig, key: Optional[str] = None
) -> str:
    if config.type == "memory":
        return base64.b64encode(file).decode("ascii")
    elif config.type == "bos":
        # TODO: Currently BOS clients are created on the fly since they are not
        # thread-safe. Should we use a background thread with a queue or use a
        # dedicated thread?
        bos_cfg = BceClientConfiguration(
            credentials=BceCredentials(
                config.ak.get_secret_value(), config.sk.get_secret_value()
            ),
            endpoint=config.endpoint,
            connection_timeout_in_mills=config.connection_timeout_in_mills,
        )
        client = BosClient(bos_cfg)
        if key is None:
            key = str(uuid.uuid4())
        if config.key_prefix:
            key = f"{config.key_prefix}{key}"
        client.put_object_from_string(bucket=config.bucket_name, key=key, data=file)
        url = client.generate_pre_signed_url(
            config.bucket_name, key, expiration_in_seconds=-1
        ).decode("ascii")
        return url
    else:
        assert_never(config.type)
