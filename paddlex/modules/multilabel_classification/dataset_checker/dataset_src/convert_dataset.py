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


import os

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.errors import ConvertFailedError
from .....utils.logging import info, warning

if is_dep_available("pycocotools"):
    from pycocotools.coco import COCO
if is_dep_available("tqdm"):
    from tqdm import tqdm


def check_src_dataset(root_dir, dataset_type):
    """check src dataset format validity"""
    if dataset_type in ("COCO"):
        pass
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 COCO 格式。"
        )

    err_msg_prefix = f"数据格式转换失败！请参考上述`{dataset_type}格式数据集示例`检查待转换数据集格式。"

    for anno in ["annotations/instance_train.json", "annotations/instance_val.json"]:
        src_anno_path = os.path.join(root_dir, anno)
        if not os.path.exists(src_anno_path):
            raise ConvertFailedError(
                message=f"{err_msg_prefix}保证{src_anno_path}文件存在。"
            )
    return None


def convert(dataset_type, input_dir):
    """convert dataset to multilabel format"""
    # check format validity
    check_src_dataset(input_dir, dataset_type)

    if dataset_type in ("COCO"):
        convert_coco_dataset(input_dir)
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 COCO 格式。"
        )


def convert_coco_dataset(root_dir):
    for anno in ["annotations/instance_train.json", "annotations/instance_val.json"]:
        src_img_dir = root_dir
        src_anno_path = os.path.join(root_dir, anno)
        coco2multilabels(src_img_dir, src_anno_path, root_dir)


@function_requires_deps("pycocotools", "tqdm")
def coco2multilabels(src_img_dir, src_anno_path, root_dir):
    image_dir = os.path.join(root_dir, "images")
    label_type = (
        os.path.basename(src_anno_path).replace("instance_", "").replace(".json", "")
    )
    anno_save_path = os.path.join(root_dir, "{}.txt".format(label_type))
    coco = COCO(src_anno_path)
    cat_id_map = {
        old_cat_id: new_cat_id for new_cat_id, old_cat_id in enumerate(coco.getCatIds())
    }
    num_classes = len(list(cat_id_map.keys()))

    with open(anno_save_path, "w") as fp:
        for img_id in tqdm(sorted(coco.getImgIds())):
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            img_filepath = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_filepath):
                warning(
                    "Illegal image file: {}, "
                    "and it will be ignored".format(img_filepath)
                )
                continue

            if img_w < 0 or img_h < 0:
                warning(
                    "Illegal width: {} or height: {} in annotation, "
                    "and im_id: {} will be ignored".format(img_w, img_h, img_id)
                )
                continue

            ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
            instances = coco.loadAnns(ins_anno_ids)

            label = [0] * num_classes
            for instance in instances:
                label[cat_id_map[instance["category_id"]]] = 1
            img_filename = os.path.join("images", img_filename)
            fp.writelines("{}\t{}\n".format(img_filename, ",".join(map(str, label))))
        fp.close()
    if label_type == "train":
        label_txt_save_path = os.path.join(root_dir, "label.txt")
        with open(label_txt_save_path, "w") as fp:
            for cat in coco.cats.values():
                id = cat["id"]
                name = cat["name"]
                fp.writelines("{} {}\n".format(id, name))
            fp.close()
            info("Save label names to {}.".format(label_txt_save_path))
