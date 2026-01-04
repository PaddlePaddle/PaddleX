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


import json
import os
import random
import shutil
import xml.etree.ElementTree as ET

from .....utils.deps import function_requires_deps, is_dep_available
from .....utils.errors import ConvertFailedError
from .....utils.file_interface import custom_open, write_json_file
from .....utils.logging import info, warning

if is_dep_available("tqdm"):
    from tqdm import tqdm


class Indexer(object):
    """Indexer"""

    def __init__(self):
        """init indexer"""
        self._map = {}
        self.idx = 0

    def get_id(self, key):
        """get id by key"""
        if key not in self._map:
            self.idx += 1
            self._map[key] = self.idx
        return self._map[key]

    def get_list(self, key_name):
        """return list containing key and id"""
        map_list = []
        for key in self._map:
            val = self._map[key]
            map_list.append({key_name: key, "id": val})
        return map_list


class Extension(object):
    """Extension"""

    def __init__(self, exts_list):
        """init extension"""
        self._exts_list = ["." + ext for ext in exts_list]

    def __iter__(self):
        """iterator"""
        return iter(self._exts_list)

    def update(self, ext):
        """update extension"""
        self._exts_list.remove(ext)
        self._exts_list.insert(0, ext)


def check_src_dataset(root_dir, dataset_type):
    """check src dataset format validity"""
    if dataset_type in ("VOC", "VOCWithUnlabeled"):
        pass
    elif dataset_type in ("LabelMe", "LabelMeWithUnlabeled"):
        pass
    else:
        raise ConvertFailedError(
            message=f"数据格式转换失败！不支持{dataset_type}格式数据集。当前仅支持 VOC、LabelMe 和 VOCWithUnlabeled、LabelMeWithUnlabeled 格式。"
        )

    err_msg_prefix = f"数据格式转换失败！请参考上述`{dataset_type}格式数据集示例`检查待转换数据集格式。"

    anno_map = {}
    for dst_anno, src_anno in [
        ("instance_train.json", "train_anno_list.txt"),
        ("instance_val.json", "val_anno_list.txt"),
    ]:
        src_anno_path = os.path.join(root_dir, src_anno)
        if not os.path.exists(src_anno_path):
            if dst_anno == "instance_train.json":
                raise ConvertFailedError(
                    message=f"{err_msg_prefix}保证{src_anno_path}文件存在。"
                )
            continue
        with custom_open(src_anno_path, "r") as f:
            anno_list = f.readlines()
        for anno_fn in anno_list:
            anno_fn = anno_fn.strip().split(" ")[-1]
            anno_path = os.path.join(root_dir, anno_fn)
            if not os.path.exists(anno_path):
                raise ConvertFailedError(
                    message=f'{err_msg_prefix}保证"{src_anno_path}"中的"{anno_fn}"文件存在。'
                )
        anno_map[dst_anno] = src_anno_path
    return anno_map


def convert(dataset_type, input_dir):
    """convert dataset to coco format"""
    # check format validity
    anno_map = check_src_dataset(input_dir, dataset_type)
    (
        convert_voc_dataset(input_dir, anno_map)
        if dataset_type in ("VOC", "VOCWithUnlabeled")
        else convert_labelme_dataset(input_dir, anno_map)
    )


def split_anno_list(root_dir, anno_map):
    """Split anno list to 80% train and 20% val"""

    train_anno_list = []
    val_anno_list = []
    anno_list_bak = os.path.join(root_dir, "train_anno_list.txt.bak")
    shutil.move(anno_map["instance_train.json"], anno_list_bak),
    with custom_open(anno_list_bak, "r") as f:
        src_anno = f.readlines()
    random.shuffle(src_anno)
    train_anno_list = src_anno[: int(len(src_anno) * 0.8)]
    val_anno_list = src_anno[int(len(src_anno) * 0.8) :]
    with custom_open(os.path.join(root_dir, "train_anno_list.txt"), "w") as f:
        f.writelines(train_anno_list)
    with custom_open(os.path.join(root_dir, "val_anno_list.txt"), "w") as f:
        f.writelines(val_anno_list)
    anno_map["instance_train.json"] = os.path.join(root_dir, "train_anno_list.txt")
    anno_map["instance_val.json"] = os.path.join(root_dir, "val_anno_list.txt")
    msg = f"{os.path.join(root_dir,'val_anno_list.txt')}不存在，数据集已默认按照80%训练集，20%验证集划分,\
        且将原始'train_anno_list.txt'重命名为'train_anno_list.txt.bak'."

    warning(msg)
    return anno_map


def convert_labelme_dataset(root_dir, anno_map):
    """convert dataset labeled by LabelMe to coco format"""
    label_indexer = Indexer()
    img_indexer = Indexer()

    annotations_dir = os.path.join(root_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # FIXME(gaotingquan): support lmssl
    unlabeled_path = os.path.join(root_dir, "unlabeled.txt")
    if os.path.exists(unlabeled_path):
        shutil.move(unlabeled_path, os.path.join(annotations_dir, "unlabeled.txt"))

    # 不存在val_anno_list，对原始数据集进行划分
    if "instance_val.json" not in anno_map:
        anno_map = split_anno_list(root_dir, anno_map)

    for dst_anno in anno_map:
        labelme2coco(
            label_indexer,
            img_indexer,
            root_dir,
            anno_map[dst_anno],
            os.path.join(annotations_dir, dst_anno),
        )


@function_requires_deps("tqdm")
def labelme2coco(label_indexer, img_indexer, root_dir, anno_path, save_path):
    """convert json files generated by LabelMe to coco format and save to files"""
    with custom_open(anno_path, "r") as f:
        json_list = f.readlines()

    anno_num = 0
    anno_list = []
    image_list = []
    info(f"Start loading json annotation files from {anno_path} ...")
    for json_path in tqdm(json_list):
        json_path = json_path.strip()
        if not json_path.endswith(".json"):
            info(f'An illegal json path("{json_path}") found! Has been ignored.')
            continue

        with custom_open(os.path.join(root_dir, json_path.strip()), "r") as f:
            labelme_data = json.load(f)

        img_id = img_indexer.get_id(labelme_data["imagePath"])
        image_list.append(
            {
                "id": img_id,
                "file_name": labelme_data["imagePath"].split("/")[-1],
                "width": labelme_data["imageWidth"],
                "height": labelme_data["imageHeight"],
            }
        )

        for shape in labelme_data["shapes"]:
            assert shape["shape_type"] == "rectangle", "Only rectangle are supported."
            category_id = label_indexer.get_id(shape["label"])
            (x1, y1), (x2, y2) = shape["points"]
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            bbox = list(map(float, [x1, y1, x2 - x1, y2 - y1]))
            anno_num += 1
            anno_list.append(
                {
                    "image_id": img_id,
                    "bbox": bbox,
                    "category_id": category_id,
                    "id": anno_num,
                    "iscrowd": 0,
                    "area": bbox[2] * bbox[3],
                    "ignore": 0,
                }
            )

    category_list = label_indexer.get_list(key_name="name")
    data_coco = {
        "images": image_list,
        "categories": category_list,
        "annotations": anno_list,
    }

    write_json_file(data_coco, save_path)
    info(f"The converted annotations has been save to {save_path}.")


def convert_voc_dataset(root_dir, anno_map):
    """convert VOC format dataset to coco format"""
    label_indexer = Indexer()
    img_indexer = Indexer()

    annotations_dir = os.path.join(root_dir, "annotations")
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    # FIXME(gaotingquan): support lmssl
    unlabeled_path = os.path.join(root_dir, "unlabeled.txt")
    if os.path.exists(unlabeled_path):
        shutil.move(unlabeled_path, os.path.join(annotations_dir, "unlabeled.txt"))

    # 不存在val_anno_list，对原始数据集进行划分
    if "instance_val.json" not in anno_map:
        anno_map = split_anno_list(root_dir, anno_map)

    for dst_anno in anno_map:
        ann_paths = voc_get_label_anno(root_dir, anno_map[dst_anno])

        voc_xmls_to_cocojson(
            root_dir=root_dir,
            annotation_paths=ann_paths,
            label_indexer=label_indexer,
            img_indexer=img_indexer,
            output=annotations_dir,
            output_file=dst_anno,
        )


def voc_get_label_anno(root_dir, anno_path):
    """
    Read VOC format annotation file.

    Args:
        root_dir (str): The directory of VOC annotation file.
        anno_path (str): The annotation file path.

    Returns:
        tuple: A tuple of two elements, the first of which is of type dict, representing the mapping between tag names
        and their corresponding ids, and the second of type list, representing the list of paths to all annotated files.
    """
    if not os.path.exists(anno_path):
        info(f"The annotation file {anno_path} don't exists, has been ignored!")
        return []
    with custom_open(anno_path, "r") as f:
        ann_ids = f.readlines()

    ann_paths = []
    info(f"Start loading xml annotation files from {anno_path} ...")
    for aid in ann_ids:
        aid = aid.strip().split(" ")[-1]
        if not aid.endswith(".xml"):
            info(f'An illegal xml path("{aid}") found! Has been ignored.')
            continue
        ann_path = os.path.join(root_dir, aid)
        ann_paths.append(ann_path)

    return ann_paths


def voc_get_image_info(annotation_root, img_indexer):
    """
    Get the image info from VOC annotation file.

    Args:
        annotation_root: The annotation root.
        img_indexer: indexer to get image id by filename.

    Returns:
        dict: The image info.

    Raises:
        AssertionError: When filename cannot be found in 'annotation_root'.
    """
    filename = annotation_root.findtext("filename")
    assert filename is not None, filename
    os.path.basename(filename)
    im_id = img_indexer.get_id(filename)

    size = annotation_root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    image_info = {"file_name": filename, "height": height, "width": width, "id": im_id}
    return image_info


def voc_get_coco_annotation(obj, label_indexer):
    """
    Convert VOC format annotation to COCO format.

    Args:
        obj: a obj in VOC.
        label_indexer: indexer to get category id by label name.

    Returns:
        dict: A dict with the COCO format annotation info.

    Raises:
        AssertionError: When the width or height of the annotation box is illegal.
    """
    label = obj.findtext("name")
    category_id = label_indexer.get_id(label)
    bndbox = obj.find("bndbox")
    xmin = float(bndbox.findtext("xmin"))
    ymin = float(bndbox.findtext("ymin"))
    xmax = float(bndbox.findtext("xmax"))
    ymax = float(bndbox.findtext("ymax"))
    if xmin > xmax or ymin > ymax:
        temp = xmin
        xmin = min(xmin, xmax)
        xmax = max(temp, xmax)
        temp = ymin
        ymin = min(ymin, ymax)
        ymax = max(temp, ymax)

    o_width = xmax - xmin
    o_height = ymax - ymin
    anno = {
        "area": o_width * o_height,
        "iscrowd": 0,
        "bbox": [xmin, ymin, o_width, o_height],
        "category_id": category_id,
        "ignore": 0,
    }
    return anno


@function_requires_deps("tqdm")
def voc_xmls_to_cocojson(
    root_dir, annotation_paths, label_indexer, img_indexer, output, output_file
):
    """
    Convert VOC format data to COCO format.

    Args:
        annotation_paths (list): A list of paths to the XML files.
        label_indexer: indexer to get category id by label name.
        img_indexer: indexer to get image id by filename.
        output (str): The directory to save output JSON file.
        output_file (str): Output JSON file name.

    Returns:
        None
    """
    extension_list = ["jpg", "png", "jpeg", "JPG", "PNG", "JPEG"]
    suffixs = Extension(extension_list)

    def match(root_dir, prefilename, prexlm_name):
        """matching extension"""
        for ext in suffixs:
            if os.path.exists(os.path.join(root_dir, "images", prefilename + ext)):
                suffixs.update(ext)
                return prefilename + ext
            elif os.path.exists(os.path.join(root_dir, "images", prexlm_name + ext)):
                suffixs.update(ext)
                return prexlm_name + ext
        return None

    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }

    bnd_id = 1  # bounding box start id
    info("Start converting !")
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()
        file_name = ann_root.find("filename")
        prefile_name = file_name.text.split(".")[0]
        prexlm_name = os.path.basename(a_path).split(".")[0]
        # 根据file_name 和 xlm_name 分别匹配查找图片
        f_name = match(root_dir, prefile_name, prexlm_name)
        if f_name is not None:
            file_name.text = f_name
        else:
            prefile_name_set = set({prefile_name, prexlm_name})
            prefile_name_set = ",".join(prefile_name_set)
            suffix_set = ",".join(extension_list)
            images_path = os.path.join(root_dir, "images")
            info(
                f"{images_path}/{{{prefile_name_set}}}.{{{suffix_set}}} both not exists,will be skipped."
            )
            continue
        img_info = voc_get_image_info(ann_root, img_indexer)
        output_json_dict["images"].append(img_info)

        for obj in ann_root.findall("object"):
            if obj.find("bndbox") is None:  # Skip the object without bndbox
                continue
            ann = voc_get_coco_annotation(obj=obj, label_indexer=label_indexer)
            ann.update({"image_id": img_info["id"], "id": bnd_id})
            output_json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    output_json_dict["categories"] = label_indexer.get_list(key_name="name")
    output_file = os.path.join(output, output_file)
    write_json_file(output_json_dict, output_file)
    info(f"The converted annotations has been save to {output_file}.")
