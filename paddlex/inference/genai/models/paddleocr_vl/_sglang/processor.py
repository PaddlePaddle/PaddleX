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


from typing import List, Union

from ......utils.deps import is_dep_available

if is_dep_available("sglang"):
    from sglang.srt.multimodal.processors.base_processor import (
        BaseMultimodalProcessor,
        MultimodalSpecialTokens,
    )

    class PPOCRVLImageProcessor(BaseMultimodalProcessor):

        def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
            super().__init__(hf_config, server_args, _processor, *args, **kwargs)

            self.mm_tokens = MultimodalSpecialTokens(
                image_token="<|vision_start|><|image_pad|><|vision_end|>",
                image_token_id=hf_config.image_token_id,
            ).build(_processor)

        async def process_mm_data_async(
            self,
            image_data: List[Union[str, bytes]],
            input_text,
            request_obj,
            *args,
            **kwargs,
        ):
            base_out = self.load_mm_data(
                prompt=input_text,
                image_data=image_data,
                multimodal_tokens=self.mm_tokens,
            )

            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                base_out, self.mm_tokens
            )

            return {
                "mm_items": mm_items,
                "input_ids": input_ids.tolist(),
                "im_token_id": self.mm_tokens.image_token_id,
            }
