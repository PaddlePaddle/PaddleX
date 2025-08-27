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

import math
import time
from typing import List

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatCompletionChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from paddlex_hps_server import BaseTritonPythonModel, logging, schemas, utils
from PIL import Image


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.doc_understanding.InferRequest

    def get_result_model_type(self):
        return ChatCompletion

    @staticmethod
    def _resize_image_with_token_limit(image, max_token_num=2200, tile_size=28):
        image = Image.fromarray(image)
        w0, h0 = image.width, image.height
        tokens = math.ceil(w0 / tile_size) * math.ceil(h0 / tile_size)
        if tokens <= max_token_num:
            return image

        k = math.sqrt(
            max_token_num / (math.ceil(w0 / tile_size) * math.ceil(h0 / tile_size))
        )
        k = min(1.0, k)
        w_new = max(int(w0 * k), tile_size)
        h_new = max(int(h0 * k), tile_size)
        new_size = (w_new, h_new)
        resized_image = image.resize(new_size)
        tokens_new = math.ceil(w_new / tile_size) * math.ceil(h_new / tile_size)
        logging.info(
            f"Resizing image from {w0}x{h0} to {w_new}x{h_new}, "
            f"which will reduce the image tokens from {tokens} to {tokens_new}."
        )

        return resized_image

    def run(self, input, log_id):
        def _process_messages(messages: List[schemas.doc_understanding.Message]):
            system_message = ""
            user_message = ""
            image_url = ""

            for msg in messages:
                if msg.role == schemas.doc_understanding.RoleType.SYSTEM:
                    if isinstance(msg.content, list):
                        for content in msg.content:
                            if isinstance(
                                content, schemas.doc_understanding.TextContent
                            ):
                                system_message = content.text
                                break
                    else:
                        system_message = msg.content

                elif msg.role == schemas.doc_understanding.RoleType.USER:
                    if isinstance(msg.content, list):
                        for content in msg.content:
                            if isinstance(content, str):
                                user_message = content
                            else:
                                if isinstance(
                                    content, schemas.doc_understanding.TextContent
                                ):
                                    user_message = content.text
                                elif isinstance(
                                    content, schemas.doc_understanding.ImageContent
                                ):
                                    image_url = content.image_url
                                    if isinstance(
                                        image_url, schemas.doc_understanding.ImageUrl
                                    ):
                                        image_url = image_url.url
                    else:
                        user_message = msg.content
            return system_message, user_message, image_url

        system_message, user_message, image_url = _process_messages(input.messages)
        if input.max_image_tokens is not None:
            if image_url.startswith("data:image"):
                _, image_url = image_url.split(",", 1)
            img_bytes = utils.get_raw_bytes(image_url)
            image = utils.image_bytes_to_array(img_bytes)
            image = self._resize_image_with_token_limit(image, input.max_image_tokens)
        else:
            image = image_url

        result = list(self.pipeline({"image": image, "query": user_message}))[0]

        return ChatCompletion(
            id=log_id,
            model=input.model,
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
