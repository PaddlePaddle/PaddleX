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

from typing import Dict, List, Union

import numpy as np
import paddle
import requests
from paddle.vision import transforms
from PIL import Image

from ....utils.benchmark import benchmark

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


class GOTImageProcessor(object):
    def __init__(self, image_size=1024):

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation="bicubic"),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class PPChart2TableProcessor(object):
    def __init__(self, image_processor, tokenizer, dtype, **kwargs):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.dtype = dtype

        prompt = (
            "<|im_start|>system\n"
            "You should follow the instructions carefully and explain your answers in detail.<|im_end|><|im_start|>user\n"
            "<img>" + "<imgpad>" * 256 + "</img>\n"
            "Chart to table<|im_end|><|im_start|>assistant\n"
        )
        self.input_ids = paddle.to_tensor(self.tokenizer([prompt]).input_ids)

    @benchmark.timeit
    def preprocess(self, image: Union[str, Image.Image, np.ndarray, Dict, List]):
        if isinstance(image, (str, Image.Image, np.ndarray)):
            image = [image]
        elif isinstance(image, dict):
            image = [image["image"]]

        assert isinstance(image, list)
        images = [
            image_["image"] if isinstance(image_, dict) else image_ for image_ in image
        ]
        images = [
            self.image_processor(self._load_image(image)).unsqueeze(0).to(self.dtype)
            for image in images
        ]
        img_cnt = len(images)

        input_ids = paddle.tile(self.input_ids, [img_cnt, 1])

        return {"input_ids": input_ids, "images": images}

    @benchmark.timeit
    def postprocess(self, model_pred, *args, **kwargs):
        return self.tokenizer.batch_decode(
            model_pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def _load_image(self, image_file):
        from io import BytesIO

        if isinstance(image_file, Image.Image):
            image = image_file.convert("RGB")
        elif isinstance(image_file, np.ndarray):
            image = Image.fromarray(image_file)
        elif image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image
