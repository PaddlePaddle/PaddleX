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

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ndarray

from ....utils.deps import class_requires_deps, function_requires_deps, is_dep_available
from ...common.reader import ReadImage as CommonReadImage
from ...utils.benchmark import benchmark
from ..common import Normalize as CommonNormalize
from ..common import Resize as CommonResize

if is_dep_available("opencv-contrib-python"):
    import cv2

Boxes = List[dict]
Number = Union[int, float]


@benchmark.timeit_with_options(name=None, is_read_operation=True)
@class_requires_deps("opencv-contrib-python")
class ReadImage(CommonReadImage):
    """Reads images from a list of raw image data or file paths."""

    def __call__(self, raw_imgs: List[Union[ndarray, str, dict]]) -> List[dict]:
        """Processes the input list of raw image data or file paths and returns a list of dictionaries containing image information.

        Args:
            raw_imgs (List[Union[ndarray, str]]): A list of raw image data (numpy ndarrays) or file paths (strings).

        Returns:
            List[dict]: A list of dictionaries, each containing image information.
        """
        out_datas = []
        for raw_img in raw_imgs:
            data = dict()
            if isinstance(raw_img, str):
                data["img_path"] = raw_img
            if isinstance(raw_img, dict):
                if "img" in raw_img:
                    src_img = raw_img["img"]
                elif "img_path" in raw_img:
                    src_img = raw_img["img_path"]
                    data["img_path"] = src_img
                else:
                    raise ValueError(
                        "When raw_img is dict, must have one of keys ['img', 'img_path']."
                    )
                data.update(raw_img)
                raw_img = src_img
            img, ori_img = self.read(raw_img)
            data["img"] = img
            data["ori_img"] = ori_img
            data["img_size"] = [img.shape[1], img.shape[0]]  # [size_w, size_h]
            data["ori_img_size"] = [img.shape[1], img.shape[0]]  # [size_w, size_h]

            out_datas.append(data)

        return out_datas

    def read(self, img):
        if isinstance(img, np.ndarray):
            ori_img = img
            if self.format == "RGB":
                img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            return img, ori_img
        elif isinstance(img, str):
            blob = self._img_reader.read(img)
            if blob is None:
                raise Exception(f"Image read Error: {img}")

            ori_img = blob
            if self.format == "RGB":
                if blob.ndim != 3:
                    raise RuntimeError("Array is not 3-dimensional.")
                # BGR to RGB
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            return blob, ori_img
        else:
            raise TypeError(
                f"ReadImage only supports the following types:\n"
                f"1. str, indicating a image file path or a directory containing image files.\n"
                f"2. numpy.ndarray.\n"
                f"However, got type: {type(img).__name__}."
            )


@benchmark.timeit
class Resize(CommonResize):
    def __call__(self, datas: List[dict]) -> List[dict]:
        """
        Args:
            datas (List[dict]): A list of dictionaries, each containing image data with key 'img'.

        Returns:
            List[dict]: A list of dictionaries with updated image data, including resized images,
                original image sizes, resized image sizes, and scale factors.
        """
        for data in datas:
            ori_img = data["img"]
            if "ori_img_size" not in data:
                data["ori_img_size"] = [ori_img.shape[1], ori_img.shape[0]]
            ori_img_size = data["ori_img_size"]

            img = self.resize(ori_img)
            data["img"] = img

            img_size = [img.shape[1], img.shape[0]]
            data["img_size"] = img_size  # [size_w, size_h]

            data["scale_factors"] = [  # [w_scale, h_scale]
                img_size[0] / ori_img_size[0],
                img_size[1] / ori_img_size[1],
            ]

        return datas


@benchmark.timeit
class Normalize(CommonNormalize):
    def __call__(self, datas: List[dict]) -> List[dict]:
        """Normalizes images in a list of dictionaries. Iterates over each dictionary,
        applies normalization to the 'img' key, and returns the modified list.
        """
        for data in datas:
            data["img"] = self.norm(data["img"])
        return datas


@benchmark.timeit
class ToCHWImage:
    """Converts images in a list of dictionaries from HWC to CHW format."""

    def __call__(self, datas: List[dict]) -> List[dict]:
        """Converts the image data in the list of dictionaries from HWC to CHW format in-place.

        Args:
            datas (List[dict]): A list of dictionaries, each containing an image tensor in 'img' key with HWC format.

        Returns:
            List[dict]: The same list of dictionaries with the image tensors converted to CHW format.
        """
        for data in datas:
            data["img"] = data["img"].transpose((2, 0, 1))
        return datas


@benchmark.timeit
class ToBatch:
    """
    Class for batch processing of data dictionaries.

    Args:
        ordered_required_keys (Optional[Tuple[str]]): A tuple of keys that need to be present in the input data dictionaries in a specific order.
    """

    def __init__(self, ordered_required_keys: Optional[Tuple[str]] = None):
        self.ordered_required_keys = ordered_required_keys

    def apply(
        self, datas: List[dict], key: str, dtype: np.dtype = np.float32
    ) -> np.ndarray:
        """
        Apply batch processing to a list of data dictionaries.

        Args:
            datas (List[dict]): A list of data dictionaries to process.
            key (str): The key in the data dictionaries to extract and batch.
            dtype (np.dtype): The desired data type of the output array (default is np.float32).

        Returns:
            np.ndarray: A numpy array containing the batched data.

        Raises:
            KeyError: If the specified key is not found in any of the data dictionaries.
        """
        if key == "img_size":
            # [h, w] size for det models
            img_sizes = [data[key][::-1] for data in datas]
            return np.stack(img_sizes, axis=0).astype(dtype=dtype, copy=False)

        elif key == "scale_factors":
            # [h, w] scale factors for det models, default [1.0, 1.0]
            scale_factors = [data.get(key, [1.0, 1.0])[::-1] for data in datas]
            return np.stack(scale_factors, axis=0).astype(dtype=dtype, copy=False)

        else:
            return np.stack([data[key] for data in datas], axis=0).astype(
                dtype=dtype, copy=False
            )

    def __call__(self, datas: List[dict]) -> Sequence[ndarray]:
        return [self.apply(datas, key) for key in self.ordered_required_keys]


@benchmark.timeit
class DetPad:
    """
    Pad image to a specified size.
    Args:
        size (list[int]): image target size
        fill_value (list[float]): rgb value of pad area, default (114.0, 114.0, 114.0)
    """

    def __init__(
        self,
        size: List[int],
        fill_value: List[Union[int, float]] = [114.0, 114.0, 114.0],
    ):
        super().__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size
        self.fill_value = fill_value

    def apply(self, img: ndarray) -> ndarray:
        im = img
        im_h, im_w = im.shape[:2]
        h, w = self.size
        if h == im_h and w == im_w:
            return im

        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = im.astype(np.float32)
        return canvas

    def __call__(self, datas: List[dict]) -> List[dict]:
        for data in datas:
            data["img"] = self.apply(data["img"])
        return datas


@benchmark.timeit
class PadStride:
    """padding image for model with FPN , instead PadBatch(pad_to_stride, pad_gt) in original config
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride: int = 0):
        super().__init__()
        self.coarsest_stride = stride

    def apply(self, img: ndarray):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
        """
        im = img
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return img
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im

    def __call__(self, datas: List[dict]) -> List[dict]:
        for data in datas:
            data["img"] = self.apply(data["img"])
        return datas


def rotate_point(pt: List[float], angle_rad: float) -> List[float]:
    """Rotate a point by an angle.
    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian
    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt


def _get_3rd_point(a: ndarray, b: ndarray) -> ndarray:
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.
    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.
    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)
    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


@function_requires_deps("opencv-contrib-python")
def get_affine_transform(
    center: ndarray,
    input_size: Union[Number, Tuple[Number, Number], ndarray],
    rot: float,
    output_size: ndarray,
    shift: Tuple[float, float] = (0.0, 0.0),
    inv: bool = False,
):
    """Get the affine transform matrix, given the center/scale/rot/output_size.
    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        input_size (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ]): Size of the destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(output_size) == 2
    assert len(shift) == 2
    if not isinstance(input_size, (ndarray, list)):
        input_size = np.array([input_size, input_size], dtype=np.float32)
    scale_tmp = input_size

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0.0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class WarpAffine:
    """Apply warp affine transformation to the image based on the given parameters.

    Args:
        keep_res (bool): Whether to keep the original resolution aspect ratio during transformation.
        pad (int): Padding value used when keep_res is True.
        input_h (int): Target height for the input image when keep_res is False.
        input_w (int): Target width for the input image when keep_res is False.
        scale (float): Scale factor for resizing.
        shift (float): Shift factor for transformation.
        down_ratio (int): Downsampling ratio for the output image.
    """

    def __init__(
        self,
        keep_res=False,
        pad=31,
        input_h=512,
        input_w=512,
        scale=0.4,
        shift=0.1,
        down_ratio=4,
    ):
        super().__init__()
        self.keep_res = keep_res
        self.pad = pad
        self.input_h = input_h
        self.input_w = input_w
        self.scale = scale
        self.shift = shift
        self.down_ratio = down_ratio

    def apply(self, img: ndarray):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        if self.keep_res:
            # True in detection eval/infer
            input_h = (h | self.pad) + 1
            input_w = (w | self.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
            c = np.array([w // 2, h // 2], dtype=np.float32)

        else:
            # False in centertrack eval_mot/eval_mot
            s = max(h, w) * 1.0
            input_h, input_w = self.input_h, self.input_w
            c = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.resize(img, (w, h))
        inp = cv2.warpAffine(
            img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR
        )

        if not self.keep_res:
            out_h = input_h // self.down_ratio
            out_w = input_w // self.down_ratio
            get_affine_transform(c, s, 0, [out_w, out_h])

        return inp

    def __call__(self, datas: List[dict]) -> List[dict]:

        for data in datas:
            ori_img = data["img"]
            if "ori_img_size" not in data:
                data["ori_img_size"] = [ori_img.shape[1], ori_img.shape[0]]

            img = self.apply(ori_img)
            data["img"] = img

        return datas


def restructured_boxes(
    boxes: ndarray, labels: List[str], img_size: Tuple[int, int]
) -> Boxes:
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    for box in boxes:
        xmin, ymin, xmax, ymax = box[2:]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [xmin, ymin, xmax, ymax],
            }
        )

    return box_list


def restructured_rotated_boxes(
    boxes: ndarray, labels: List[str], img_size: Tuple[int, int]
) -> Boxes:
    """
    Restructure the given rotated bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of rotated bounding boxes with each box represented as [cls_id, score, x1, y1, x2, y2, x3, y3, x4, y4].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    assert boxes.shape[1] == 10, "The shape of rotated boxes should be [N, 10]"
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box[2:]
        x1 = min(max(0, x1), w)
        y1 = min(max(0, y1), h)
        x2 = min(max(0, x2), w)
        y2 = min(max(0, y2), h)
        x3 = min(max(0, x3), w)
        y3 = min(max(0, y3), h)
        x4 = min(max(0, x4), w)
        y4 = min(max(0, y4), h)
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [x1, y1, x2, y2, x3, y3, x4, y4],
            }
        )

    return box_list


def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding boxes from (x1, y1, x2, y2) format using an unclipping ratio.

    Parameters:
    - boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    - unclip_ratio: tuple of (width_ratio, height_ratio), optional.

    Returns:
    - expanded_boxes: np.ndarray of shape (N, 4), where each row is (x1, y1, x2, y2).
    """
    if unclip_ratio is None:
        return boxes

    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]

                width = x2 - x1
                height = y2 - y1

                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                expanded_boxes.append([class_id, score, new_x1, new_y1, new_x2, new_y2])
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)

    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]

        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        return expanded_boxes


def iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the intersection coordinates
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)

    # Compute the area of intersection
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    # Compute the IoU
    iou_value = inter_area / float(box1_area + box2_area - inter_area)

    return iou_value


def nms(boxes, iou_same=0.6, iou_diff=0.95):
    """Perform Non-Maximum Suppression (NMS) with different IoU thresholds for same and different classes."""
    # Extract class scores
    scores = boxes[:, 1]

    # Sort indices by scores in descending order
    indices = np.argsort(scores)[::-1]
    selected_boxes = []

    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_box[1]
        current_coords = current_box[2:]

        selected_boxes.append(current)
        indices = indices[1:]

        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff

            # If the IoU is below the threshold, keep the box
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def is_contained(box1, box2):
    """Check if box1 is contained within box2."""
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    iou = intersect_area / box1_area if box1_area > 0 else 0
    return iou >= 0.9


def check_containment(boxes, formula_index=None, category_index=None, mode=None):
    """Check containment relationships among boxes."""
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if formula_index is not None:
                if boxes[i][0] == formula_index and boxes[j][0] != formula_index:
                    continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


@benchmark.timeit
class DetPostProcess:
    """Save Result Transform

    This class is responsible for post-processing detection results, including
    thresholding, non-maximum suppression (NMS), and restructuring the boxes
    based on the input type (normal or rotated object detection).
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        """Initialize the DetPostProcess class.

        Args:
            threshold (float, optional): The threshold to apply to the detection scores. Defaults to 0.5.
            labels (Optional[List[str]], optional): The list of labels for the detection categories. Defaults to None.
            layout_postprocess (bool, optional): Whether to apply layout post-processing. Defaults to False.
        """
        super().__init__()
        self.labels = labels

    def apply(
        self,
        boxes: ndarray,
        img_size: Tuple[int, int],
        threshold: Union[float, dict],
        layout_nms: Optional[bool],
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        layout_merge_bboxes_mode: Optional[Union[str, dict]],
    ) -> Boxes:
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        if isinstance(threshold, float):
            expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
        elif isinstance(threshold, dict):
            category_filtered_boxes = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_threshold = threshold.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] > category_threshold) & (
                    category_boxes[:, 0] > -1
                )
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )

        if layout_nms:
            selected_indices = nms(boxes, iou_same=0.6, iou_diff=0.98)
            boxes = np.array(boxes[selected_indices])

        filter_large_image = True
        if filter_large_image and len(boxes) > 1 and boxes.shape[1] == 6:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = self.labels.index("image") if "image" in self.labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes:
                label_index, score, xmin, ymin, xmax, ymax = box
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if len(filtered_boxes) == 0:
                filtered_boxes = boxes
            boxes = np.array(filtered_boxes)

        if layout_merge_bboxes_mode:
            formula_index = (
                self.labels.index("formula") if "formula" in self.labels else None
            )
            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"

                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    contains_other, contained_by_other = check_containment(
                        boxes, formula_index
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes = boxes[contained_by_other == 0]
                    elif layout_merge_bboxes_mode == "small":
                        boxes = boxes[(contains_other == 0) | (contained_by_other == 1)]
            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in [
                        "union",
                        "large",
                        "small",
                    ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            contains_other, contained_by_other = check_containment(
                                boxes, formula_index, category_index, mode=layout_mode
                            )
                            # Remove boxes that are contained by other boxes
                            keep_mask &= contained_by_other == 0
                        elif layout_mode == "small":
                            contains_other, contained_by_other = check_containment(
                                boxes, formula_index, category_index, mode=layout_mode
                            )
                            # Keep boxes that do not contain others or are contained by others
                            keep_mask &= (contains_other == 0) | (
                                contained_by_other == 1
                            )
                boxes = boxes[keep_mask]

        if boxes.size == 0:
            return []

        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), f"The length of `layout_unclip_ratio` should be 2."
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or  Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
                )
            boxes = unclip_boxes(boxes, layout_unclip_ratio)

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(boxes, self.labels, img_size)
        elif boxes.shape[1] == 10:
            """Adapt For Rotated Object Detection"""
            boxes = restructured_rotated_boxes(boxes, self.labels, img_size)
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

    def __call__(
        self,
        batch_outputs: List[dict],
        datas: List[dict],
        threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
    ) -> List[Boxes]:
        """Apply the post-processing to a batch of outputs.

        Args:
            batch_outputs (List[dict]): The list of detection outputs.
            datas (List[dict]): The list of input data.

        Returns:
            List[Boxes]: The list of post-processed detection boxes.
        """
        outputs = []
        for data, output in zip(datas, batch_outputs):
            boxes = self.apply(
                output["boxes"],
                data["ori_img_size"],
                threshold,
                layout_nms,
                layout_unclip_ratio,
                layout_merge_bboxes_mode,
            )
            outputs.append(boxes)
        return outputs
