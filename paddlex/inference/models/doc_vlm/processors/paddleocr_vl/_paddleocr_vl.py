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

# This file is based on https://github.com/Kwai-Keye/Keye/blob/main/keye-vl-8b-preview/processing_keye.py
# Original header:
# Copyright 2025 The Keye Team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import List

import paddle

from .....utils.benchmark import benchmark
from ..common import BatchFeature, fetch_image


class PaddleOCRVLProcessor(object):
    _DEFAULT_TEXT_KWARGS = {
        "padding": False,
        "return_tensors": "pd",
    }
    _DEFAULT_VIDEO_KWARGS = {
        "fps": 2.0,
        "return_tensors": "pd",
    }

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
    ):
        self.image_token = (
            "<|IMAGE_PLACEHOLDER|>"
            if not hasattr(tokenizer, "image_token")
            else tokenizer.image_token
        )
        self.video_token = (
            "<|video_pad|>"
            if not hasattr(tokenizer, "video_token")
            else tokenizer.video_token
        )
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    @benchmark.timeit
    def preprocess(
        self,
        input_dicts,
    ):
        images = [fetch_image(input_dict["image"]) for input_dict in input_dicts]

        text = []
        for input_dict in input_dicts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},  # placeholder
                        {"type": "text", "text": input_dict["query"]},
                    ],
                }
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            text.append(prompt)

        videos = None
        output_kwargs = {
            "tokenizer_init_kwargs": self.tokenizer.init_kwargs,
            "text_kwargs": copy.deepcopy(self._DEFAULT_TEXT_KWARGS),
            "video_kwargs": copy.deepcopy(self._DEFAULT_VIDEO_KWARGS),
        }

        if images is not None:
            image_inputs = self.image_processor(images=images, return_tensors="pd")
            image_inputs["pixel_values"] = image_inputs["pixel_values"]
            image_grid_thw = image_inputs["image_grid_thw"]

        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            # TODO: add video processing
            videos_inputs = self.image_processor(
                images=None, videos=videos, **output_kwargs["images_kwargs"]
            )
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / fps
                ] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [
                    self.image_processor.temporal_patch_size / tmp for tmp in fps
                ]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update(
                {"second_per_grid_ts": paddle.to_tensor(second_per_grid_ts)}
            )

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>"
                        * int(
                            image_grid_thw[index].prod()
                            // self.image_processor.merge_size
                            // self.image_processor.merge_size
                        ),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>"
                        * (
                            video_grid_thw[index].prod()
                            // self.image_processor.merge_size
                            // self.image_processor.merge_size
                        ),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})

    @benchmark.timeit
    def postprocess(self, model_pred, **kwargs) -> List[str]:
        return self.tokenizer.batch_decode(
            model_pred[0],
            skip_special_tokens=kwargs.get("skip_special_tokens", True),
            spaces_between_special_tokens=False,
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to Qwen2TokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(
            dict.fromkeys(tokenizer_input_names + image_processor_input_names)
        )
        return names_from_processor + ["second_per_grid_ts"]
