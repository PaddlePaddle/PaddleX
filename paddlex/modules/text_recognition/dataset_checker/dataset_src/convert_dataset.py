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


import math
import os
import pickle
from collections import defaultdict

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.errors import ConvertFailedError

if is_dep_available("imagesize"):
    import imagesize
if is_dep_available("tqdm"):
    from tqdm import tqdm


def check_src_dataset(root_dir, dataset_type):
    """check src dataset format validity"""
    if dataset_type in ("MSTextRecDataset"):
        pass
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 MSTextRecDataset 格式。"
        )

    err_msg_prefix = f"数据格式转换失败！请参考上述`{dataset_type}格式数据集示例`检查待转换数据集格式。"

    for anno in ["train.txt", "val.txt", "latex_ocr_tokenizer.json"]:
        src_anno_path = os.path.join(root_dir, anno)
        if not os.path.exists(src_anno_path):
            raise ConvertFailedError(
                message=f"{err_msg_prefix}保证{src_anno_path}文件存在。"
            )
    return None


def convert(dataset_type, input_dir):
    """convert dataset to pkl format"""
    # check format validity
    check_src_dataset(input_dir, dataset_type)
    if dataset_type in ("MSTextRecDataset"):
        convert_pkl_dataset(input_dir)
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 MSTextRecDataset 格式。"
        )


def convert_pkl_dataset(root_dir):
    for anno in ["train.txt", "val.txt"]:
        src_img_dir = root_dir
        src_anno_path = os.path.join(root_dir, anno)
        txt2pickle(src_img_dir, src_anno_path, root_dir)


@function_requires_deps("tqdm", "imagesize")
def txt2pickle(images, equations, save_dir):
    phase = os.path.basename(equations).replace(".txt", "")
    save_p = os.path.join(save_dir, "latexocr_{}.pkl".format(phase))
    min_dimensions = (32, 32)
    max_dimensions = (672, 192)
    data = defaultdict(lambda: [])
    pic_num = 0
    if images is not None and equations is not None:
        with open(equations, "r") as f:
            lines = f.readlines()
            for l in tqdm(lines, total=len(lines)):
                l = l.strip()
                img_name, equation = l.split("\t")
                img_path = os.path.join(images, img_name)
                width, height = imagesize.get(img_path)
                if (
                    min_dimensions[0] <= width <= max_dimensions[0]
                    and min_dimensions[1] <= height <= max_dimensions[1]
                ):
                    divide_h = math.ceil(height / 16) * 16
                    divide_w = math.ceil(width / 16) * 16
                    data[(divide_w, divide_h)].append((equation, img_name))
                    pic_num += 1
        data = dict(data)
        with open(save_p, "wb") as file:
            pickle.dump(data, file)
