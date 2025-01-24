# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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
import os.path as osp
from typing import List, Sequence, Union, Optional, Tuple

import numpy as np
import cv2
import lazy_paddle as paddle


class ResizeVideo:
    """Resizes frames of a video to a specified target size.

    This class provides functionality to resize each frame of a video to
    a specified square dimension (height and width are equal).

    Attributes:
        target_size (int): The desired size (in pixels) for both the height
            and width of each frame in the video.
    """

    def __init__(self, target_size: int = 224) -> None:
        """Initializes the ResizeVideo with a target size.

        Args:
            target_size (int): The desired size in pixels for the output
                frames. Defaults to 224.
        """
        super().__init__()
        self.target_size = target_size

    def resize(self, video: List) -> List:
        """Resizes all frames of a single video.

        Args:
            video (list): A list of segments, where each segment is a list
                of frames represented as numpy arrays.

        Returns:
            list: The input video with each frame resized to the target size.

        Raises:
            NotImplementedError: If a frame is not an instance of numpy.ndarray.
        """
        num_seg = len(video)
        seg_len = len(video[0])

        for i in range(num_seg):
            for j in range(seg_len):
                img = video[i][j]
                if isinstance(img, np.ndarray):
                    h, w, _ = img.shape
                else:
                    raise NotImplementedError(
                        "Currently, only numpy.ndarray frames are supported."
                    )
                video[i][j] = cv2.resize(
                    img,
                    (self.target_size, self.target_size),
                    interpolation=cv2.INTER_LINEAR,
                )
        return video

    def __call__(self, videos: List) -> List:
        """Resizes frames of multiple videos.

        Args:
            videos (list): A list containing multiple videos, where each video
                is a list of segments, and each segment is a list of frames.

        Returns:
            list: A list of videos with each frame resized to the target size.
        """
        return [self.resize(video) for video in videos]


class Image2Array:
    """Convert a sequence of images to a numpy array with optional transposition."""

    def __init__(self, data_format: str = "tchw") -> None:
        """
        Initializes the Image2Array class.

        Args:
            data_format (str): The format to transpose to, either 'tchw' or 'cthw'.

        Raises:
            AssertionError: If data_format is not one of the allowed values.
        """
        super().__init__()
        assert data_format in [
            "tchw",
            "cthw",
        ], f"Target format must be in ['tchw', 'cthw'], but got {data_format}"
        self.data_format = data_format

    def img2array(self, video: List) -> List:
        """
        Converts a list of video frames to a numpy array, with frames transposed.

        Args:
            video (List): A list of frames represented as numpy arrays.

        Returns:
            List: A numpy array with the video frames transposed and concatenated.
        """
        # Transpose each image from HWC to CHW format
        num_seg = len(video)
        for i in range(num_seg):
            video_one = video[i]
            video_one = [img.transpose([2, 0, 1]) for img in video_one]
            video_one = np.concatenate(
                [np.expand_dims(img, axis=1) for img in video_one], axis=1
            )
            video[i] = video_one
        return video

    def __call__(self, videos: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Process videos by converting each video to a transposed numpy array.

        Args:
            videos (List[List[np.ndarray]]): A list of videos, where each video is a list
                of frames represented as numpy arrays.

        Returns:
            List[np.ndarray]: A list of processed videos with transposed frames.
        """
        return [self.img2array(video) for video in videos]


class NormalizeVideo:
    """
    A class to normalize video frames by scaling the pixel values.
    """

    def __init__(self, scale: float = 255.0) -> None:
        """
        Initializes the NormalizeVideo class.

        Args:
            scale (float): The scale factor to normalize the frames, usually the max pixel value.
        """
        super().__init__()
        self.scale = scale

    def normalize_video(self, video: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normalizes a sequence of images by scaling the pixel values.

        Args:
            video (List[np.ndarray]): A list of frames, where each frame is a numpy array to be normalized.

        Returns:
            List[np.ndarray]: The normalized video frames as a list of numpy arrays.
        """
        num_seg = len(video)  # Number of frames in the video
        for i in range(num_seg):
            # Convert frame to float32 and scale pixel values
            video[i] = video[i].astype(np.float32) / self.scale
            # Expand dimensions if needed
            video[i] = np.expand_dims(video[i], axis=0)

        return video

    def __call__(self, videos: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """
        Apply normalization to a list of videos.

        Args:
            videos (List[List[np.ndarray]]): A list of videos, where each video is a list of frames
                represented as numpy arrays.

        Returns:
            List[List[np.ndarray]]: A list of normalized videos, each represented as a list of normalized frames.
        """
        return [self.normalize_video(video) for video in videos]


def convert2cpu(gpu_matrix):
    float_32_g = gpu_matrix.astype("float32")
    return float_32_g.cpu()


def convert2cpu_long(gpu_matrix):
    int_64_g = gpu_matrix.astype("int64")
    return int_64_g.cpu()


def get_region_boxes(
    output,
    conf_thresh=0.005,
    num_classes=24,
    anchors=[
        0.70458,
        1.18803,
        1.26654,
        2.55121,
        1.59382,
        4.08321,
        2.30548,
        4.94180,
        3.52332,
        5.91979,
    ],
    num_anchors=5,
    only_objectness=1,
):
    """
    Processes the output of a neural network to extract bounding box predictions.

    Args:
        output (Tensor): The output tensor from the neural network.
        conf_thresh (float): The confidence threshold for filtering predictions. Default is 0.005.
        num_classes (int): The number of classes for classification. Default is 24.
        anchors (List[float]): A list of anchor box dimensions used in the model. Default is a list
            of 10 predefined anchor values.
        num_anchors (int): The number of anchor boxes used in the model. Default is 5.
        only_objectness (int): If set to 1, only objectness scores are considered for filtering. Default is 1.
    Returns:
        all_box(List[List[float]]): A list of predicted bounding boxes for each image in the batch.
    """
    anchor_step = len(anchors) // num_anchors
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.shape[0]
    assert output.shape[1] == (5 + num_classes) * num_anchors
    h = output.shape[2]
    w = output.shape[3]
    all_boxes = []
    output = paddle.reshape(output, [batch * num_anchors, 5 + num_classes, h * w])
    output = paddle.transpose(output, (1, 0, 2))
    output = paddle.reshape(output, [5 + num_classes, batch * num_anchors * h * w])

    grid_x = paddle.linspace(0, w - 1, w)
    grid_x = paddle.tile(grid_x, [h, 1])
    grid_x = paddle.tile(grid_x, [batch * num_anchors, 1, 1])
    grid_x = paddle.reshape(grid_x, [batch * num_anchors * h * w]).cuda()

    grid_y = paddle.linspace(0, h - 1, h)
    grid_y = paddle.tile(grid_y, [w, 1]).t()
    grid_y = paddle.tile(grid_y, [batch * num_anchors, 1, 1])
    grid_y = paddle.reshape(grid_y, [batch * num_anchors * h * w]).cuda()

    sigmoid = paddle.nn.Sigmoid()
    xs = sigmoid(output[0]) + grid_x
    ys = sigmoid(output[1]) + grid_y

    anchor_w = paddle.to_tensor(anchors)
    anchor_w = paddle.reshape(anchor_w, [num_anchors, anchor_step])
    anchor_w = paddle.index_select(
        anchor_w, index=paddle.to_tensor(np.array([0]).astype("int32")), axis=1
    )

    anchor_h = paddle.to_tensor(anchors)
    anchor_h = paddle.reshape(anchor_h, [num_anchors, anchor_step])
    anchor_h = paddle.index_select(
        anchor_h, index=paddle.to_tensor(np.array([1]).astype("int32")), axis=1
    )

    anchor_w = paddle.tile(anchor_w, [batch, 1])
    anchor_w = paddle.tile(anchor_w, [1, 1, h * w])
    anchor_w = paddle.reshape(anchor_w, [batch * num_anchors * h * w]).cuda()

    anchor_h = paddle.tile(anchor_h, [batch, 1])
    anchor_h = paddle.tile(anchor_h, [1, 1, h * w])
    anchor_h = paddle.reshape(anchor_h, [batch * num_anchors * h * w]).cuda()

    ws = paddle.exp(output[2]) * anchor_w
    hs = paddle.exp(output[3]) * anchor_h

    det_confs = sigmoid(output[4])

    cls_confs = paddle.to_tensor(output[5 : 5 + num_classes], stop_gradient=True)
    cls_confs = paddle.transpose(cls_confs, [1, 0])
    s = paddle.nn.Softmax()
    cls_confs = paddle.to_tensor(s(cls_confs))

    cls_max_confs = paddle.max(cls_confs, axis=1)
    cls_max_ids = paddle.argmax(cls_confs, axis=1)

    cls_max_confs = paddle.reshape(cls_max_confs, [-1])
    cls_max_ids = paddle.reshape(cls_max_ids, [-1])

    sz_hw = h * w
    sz_hwa = sz_hw * num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b * sz_hwa + i * sz_hw + cy * w + cx
                    det_conf = det_confs[ind]
                    if only_objectness:
                        conf = det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]

                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        box = [
                            bcx / w,
                            bcy / h,
                            bw / w,
                            bh / h,
                            det_conf,
                            cls_max_conf,
                            cls_max_id,
                        ]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes


def nms(boxes, nms_thresh):
    """
    Performs non-maximum suppression on the input boxes based on their IoUs.
    """
    if len(boxes) == 0:
        return boxes
    det_confs = paddle.zeros([len(boxes)])
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]

    sortIds = paddle.argsort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the Intersection over Union (IoU) of two bounding boxes.
    """
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0] - box1[2] / 2.0), float(box2[0] - box2[2] / 2.0))
        Mx = max(float(box1[0] + box1[2] / 2.0), float(box2[0] + box2[2] / 2.0))
        my = min(float(box1[1] - box1[3] / 2.0), float(box2[1] - box2[3] / 2.0))
        My = max(float(box1[1] + box1[3] / 2.0), float(box2[1] + box2[3] / 2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return paddle.to_tensor(0.0)

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


class DetVideoPostProcess:
    """
    A class used to perform post-processing on detection results in videos.
    """

    def __init__(
        self,
        label_list: List[str] = [],
    ) -> None:
        """
        Args:
            labels : List[str]
                A list of labels or class names associated with the detection results.
        """
        super().__init__()
        self.labels = label_list

    def postprocess(self, pred: List, nms_thresh: float, score_thresh: float) -> List:
        font = cv2.FONT_HERSHEY_SIMPLEX
        num_seg = len(pred)
        pred_all = []
        for i in range(num_seg):
            outputs = pred[i]
            for out in outputs:
                preds = []
                out = paddle.to_tensor(out)
                all_boxes = get_region_boxes(out, num_classes=len(self.labels))
                for i in range(out.shape[0]):
                    boxes = all_boxes[i]
                    boxes = nms(boxes, nms_thresh)
                    for box in boxes:
                        x1 = round(float(box[0] - box[2] / 2.0) * 320.0)
                        y1 = round(float(box[1] - box[3] / 2.0) * 240.0)
                        x2 = round(float(box[0] + box[2] / 2.0) * 320.0)
                        y2 = round(float(box[1] + box[3] / 2.0) * 240.0)

                        det_conf = float(box[4])
                        for j in range((len(box) - 5) // 2):
                            cls_conf = float(box[5 + 2 * j].item())
                            prob = det_conf * cls_conf
                        if prob > score_thresh:
                            preds.append(
                                [[x1, y1, x2, y2], prob, self.labels[int(box[6])]]
                            )
            pred_all.append(preds)
        return pred_all

    def __call__(self, preds: List, nms_thresh, score_thresh) -> List:
        return [self.postprocess(pred, nms_thresh, score_thresh) for pred in preds]
