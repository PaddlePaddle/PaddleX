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

import time
from typing import Any, List

from fastapi import FastAPI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import AIStudioResultResponse
from ...schemas.pp_docbee import (
    INFER_ENDPOINT,
    ContentType,
    InferRequest,
    Message,
    RoleType,
)
from .._app import create_app, primary_operation


def process_messages(messages: List[Message]):
    system_message = ""
    user_message = ""
    image_url = ""

    for msg in messages:
        if msg.role == RoleType.SYSTEM:
            if isinstance(msg.content, list):
                for content in msg.content:
                    if isinstance(content, dict) and content.get("text"):
                        system_message = content["text"]
                        break
            else:
                system_message = msg.content

        elif msg.role == RoleType.USER:
            if isinstance(msg.content, list):
                for content in msg.content:
                    content_type = content.get("type")

                    if content_type == ContentType.TEXT:
                        user_message = content["text"]
                    elif content_type == ContentType.IMAGE_URL:
                        image_url = content["url"]
            else:
                user_message = msg.content
    return system_message, user_message, image_url


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> AIStudioResultResponse[ChatCompletion]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        system_message, user_message, image_url = process_messages(request.messages)
        file_bytes = await serving_utils.get_raw_bytes_async(image_url, aiohttp_session)
        image = serving_utils.image_bytes_to_array(file_bytes)

        result = (
            await pipeline.infer(
                image,
                det_threshold=request.detThreshold,
            )
        )[0]

        id = serving_utils.generate_log_id()
        return AIStudioResultResponse[ChatCompletion](
            logId=id,
            result=ChatCompletion(
                id=f"chatcmpl-{id}",
                model=InferRequest.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=ChatCompletionMessage(
                            role=RoleType.ASSISTANT,
                            content=result["result"],
                            finish_reason="stop",
                        ),
                    )
                ],
                created=int(time.time()),
                object="chat.completion",
            ),
        )

    return app
