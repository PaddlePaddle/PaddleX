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

from .....utils.deps import function_requires_deps, is_dep_available
from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...schemas.doc_understanding import (
    INFER_ENDPOINT,
    ImageContent,
    ImageUrl,
    InferRequest,
    Message,
    RoleType,
    TextContent,
)
from .._app import create_app, primary_operation

if is_dep_available("fastapi"):
    from fastapi import FastAPI
if is_dep_available("openai"):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage


@function_requires_deps("fastapi", "openai")
def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> "FastAPI":
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    @primary_operation(
        app,
        "/chat/completions",
        "inferA",
    )
    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> "ChatCompletion":
        pipeline = ctx.pipeline

        def _process_messages(messages: List[Message]):
            system_message = ""
            user_message = ""
            image_url = ""

            for msg in messages:
                if msg.role == RoleType.SYSTEM:
                    if isinstance(msg.content, list):
                        for content in msg.content:
                            if isinstance(content, TextContent):
                                system_message = content.text
                                break
                    else:
                        system_message = msg.content

                elif msg.role == RoleType.USER:
                    if isinstance(msg.content, list):
                        for content in msg.content:
                            if isinstance(content, str):
                                user_message = content
                            else:
                                if isinstance(content, TextContent):
                                    user_message = content.text
                                elif isinstance(content, ImageContent):
                                    image_url = content.image_url
                                    if isinstance(image_url, ImageUrl):
                                        image_url = image_url.url
                    else:
                        user_message = msg.content
            return system_message, user_message, image_url

        system_message, user_message, image_url = _process_messages(request.messages)
        result = (
            await pipeline.infer(
                {"image": image_url, "query": user_message},
            )
        )[0]

        return ChatCompletion(
            id=serving_utils.generate_log_id(),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=result["result"],
                    ),
                )
            ],
            created=int(time.time()),
            object="chat.completion",
        )

    return app
