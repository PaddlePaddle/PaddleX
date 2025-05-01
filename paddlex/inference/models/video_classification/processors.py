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

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark

if is_dep_available("opencv-contrib-python"):
    import cv2


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class Scale:
    """Scale images."""

    def __init__(
        self,
        short_size: int,
        fixed_ratio: bool = True,
        keep_ratio: Union[bool, None] = None,
        do_round: bool = False,
    ) -> None:
        """
        Initializes the Scale class.

        Args:
            short_size (int): The target size for the shorter side of the image.
            fixed_ratio (bool): Whether to maintain a fixed aspect ratio of 4:3.
            keep_ratio (Union[bool, None]): Whether to keep the aspect ratio. Cannot be True if fixed_ratio is True.
            do_round (bool): Whether to round the scaling factor.
        """
        super().__init__()
        self.short_size = short_size
        assert (fixed_ratio and not keep_ratio) or (
            not fixed_ratio
        ), f"fixed_ratio and keep_ratio cannot be true at the same time"
        self.fixed_ratio = fixed_ratio
        self.keep_ratio = keep_ratio
        self.do_round = do_round

    def scale(self, video: List[np.ndarray]) -> List[np.ndarray]:
        """
        Performs resize operations on a sequence of images.

        Args:
            video (List[np.ndarray]): List where each item is an image,  as a numpy array.
             For example, [np.ndarray0, np.ndarray1, np.ndarray2, ...]

        Returns:
            List[np.ndarray]: List where each item is a np.ndarray after scaling.
        """

        imgs = video

        resized_imgs = []
        for i in range(len(imgs)):
            img = imgs[i]
            if isinstance(img, np.ndarray):
                h, w, _ = img.shape
            else:
                raise NotImplementedError
            if (w <= h and w == self.short_size) or (h <= w and h == self.short_size):
                resized_imgs.append(img)
                continue

            if w <= h:
                ow = self.short_size
                if self.fixed_ratio:
                    oh = int(self.short_size * 4.0 / 3.0)
                elif self.keep_ratio is False:
                    oh = self.short_size
                else:
                    scale_factor = self.short_size / w
                    oh = (
                        int(h * float(scale_factor) + 0.5)
                        if self.do_round
                        else int(h * self.short_size / w)
                    )
                    ow = (
                        int(w * float(scale_factor) + 0.5)
                        if self.do_round
                        else self.short_size
                    )
            else:
                oh = self.short_size
                if self.fixed_ratio:
                    ow = int(self.short_size * 4.0 / 3.0)
                elif self.keep_ratio is False:
                    ow = self.short_size
                else:
                    scale_factor = self.short_size / h
                    oh = (
                        int(h * float(scale_factor) + 0.5)
                        if self.do_round
                        else self.short_size
                    )
                    ow = (
                        int(w * float(scale_factor) + 0.5)
                        if self.do_round
                        else int(w * self.short_size / h)
                    )
            resized_imgs.append(
                cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            )
        imgs = resized_imgs
        return imgs

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the scaling operation to a list of videos.

        Args:
            videos (List[np.ndarray]): A list of videos, where each video is a sequence
            of images.

        Returns:
            List[np.ndarray]: A list of videos after scaling, where each video is a list of images.
        """
        return [self.scale(video) for video in videos]


@benchmark.timeit
class CenterCrop:
    """Center crop images."""

    def __init__(self, target_size: int, do_round: bool = True) -> None:
        """
        Initializes the CenterCrop class.

        Args:
            target_size (int): The size of the cropped area.
            do_round (bool): Whether to round the crop coordinates.
        """
        super().__init__()
        self.target_size = target_size
        self.do_round = do_round

    def center_crop(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Performs center crop operations on images.

        Args:
            imgs (List[np.ndarray]): A sequence of images (a numpy array).

        Returns:
            List[np.ndarray]: A list of images after center cropping or a cropped numpy array.
        """

        crop_imgs = []
        th, tw = self.target_size, self.target_size
        for img in imgs:
            h, w, _ = img.shape
            assert (w >= self.target_size) and (
                h >= self.target_size
            ), "image width({}) and height({}) should be larger than crop size".format(
                w, h, self.target_size
            )
            x1 = int(round((w - tw) / 2.0)) if self.do_round else (w - tw) // 2
            y1 = int(round((h - th) / 2.0)) if self.do_round else (h - th) // 2
            crop_imgs.append(img[y1 : y1 + th, x1 : x1 + tw])
        return crop_imgs

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the center crop operation to a list of videos.

        Args:
            videos (List[np.ndarray]): A list of videos, where each video is a sequence of images.

        Returns:
            List[np.ndarray]: A list of videos after center cropping.
        """
        return [self.center_crop(video) for video in videos]


@benchmark.timeit
class Image2Array:
    """Convert a sequence of images to a numpy array with optional transposition."""

    def __init__(self, transpose: bool = True, data_format: str = "tchw") -> None:
        """
        Initializes the Image2Array class.

        Args:
            transpose (bool): Whether to transpose the resulting numpy array.
            data_format (str): The format to transpose to, either 'tchw' or 'cthw'.

        Raises:
            AssertionError: If data_format is not one of the allowed values.
        """
        super().__init__()
        assert data_format in [
            "tchw",
            "cthw",
        ], f"Target format must in ['tchw', 'cthw'], but got {data_format}"
        self.transpose = transpose
        self.data_format = data_format

    def img2array(self, imgs: List[np.ndarray]) -> np.ndarray:
        """
        Converts a sequence of images to a numpy array and optionally transposes it.

        Args:
            imgs (List[np.ndarray]): A list of images to be converted to a numpy array.

        Returns:
            np.ndarray: A numpy array representation of the images.
        """
        t_imgs = np.stack(imgs).astype("float32")
        if self.transpose:
            if self.data_format == "tchw":
                t_imgs = t_imgs.transpose([0, 3, 1, 2])  # tchw
            else:
                t_imgs = t_imgs.transpose([3, 0, 1, 2])  # cthw
        return t_imgs

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply the image to array conversion to a list of videos.

        Args:
            videos (List[Sequence[np.ndarray]]): A list of videos, where each video is a sequence of images.

        Returns:
            List[np.ndarray]: A list of numpy arrays, one for each video.
        """
        return [self.img2array(video) for video in videos]


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class NormalizeVideo:
    """
    Normalize video frames by subtracting the mean and dividing by the standard deviation.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        tensor_shape: Sequence[int] = [3, 1, 1],
        inplace: bool = False,
    ) -> None:
        """
        Initializes the NormalizeVideo class.

        Args:
            mean (Sequence[float]): The mean values for each channel.
            std (Sequence[float]): The standard deviation values for each channel.
            tensor_shape (Sequence[int]): The shape of the mean and std tensors.
            inplace (bool): Whether to perform normalization in place.
        """
        super().__init__()

        self.inplace = inplace
        if not inplace:
            self.mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
            self.std = np.array(std).reshape(tensor_shape).astype(np.float32)
        else:
            self.mean = np.array(mean, dtype=np.float32)
            self.std = np.array(std, dtype=np.float32)

    def normalize_video(self, imgs: np.ndarray) -> np.ndarray:
        """
        Normalizes a sequence of images.

        Args:
            imgs (np.ndarray): A numpy array of images to be normalized.

        Returns:
            np.ndarray: The normalized images as a numpy array.
        """

        if self.inplace:
            n = len(imgs)
            h, w, c = imgs[0].shape
            norm_imgs = np.empty((n, h, w, c), dtype=np.float32)
            for i, img in enumerate(imgs):
                norm_imgs[i] = img

            for img in norm_imgs:  # [n,h,w,c]
                mean = np.float64(self.mean.reshape(1, -1))  # [1, 3]
                stdinv = 1 / np.float64(self.std.reshape(1, -1))  # [1, 3]
                cv2.subtract(img, mean, img)
                cv2.multiply(img, stdinv, img)
        else:
            imgs = imgs
            norm_imgs = imgs / 255.0
            norm_imgs -= self.mean
            norm_imgs /= self.std

        imgs = norm_imgs
        imgs = np.expand_dims(imgs, axis=0).copy()
        return imgs

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply normalization to a list of videos.

        Args:
            videos (List[np.ndarray]): A list of videos, where each video is a numpy array of images.

        Returns:
            List[np.ndarray]: A list of normalized videos as numpy arrays.
        """
        return [self.normalize_video(video) for video in videos]


@benchmark.timeit
class VideoClasTopk:
    """Applies a top-k transformation on video classification predictions."""

    def __init__(self, class_ids: Optional[Sequence[Union[str, int]]] = None) -> None:
        """
        Initializes the VideoClasTopk class.

        Args:
            class_ids (Optional[Sequence[Union[str, int]]]): A list of class labels corresponding to class indices.
        """
        super().__init__()
        self.class_id_map = self._parse_class_id_map(class_ids)

    def softmax(self, data: np.ndarray) -> np.ndarray:
        """
        Applies the softmax function to an array of data.

        Args:
            data (np.ndarray): An array of data for which to compute softmax.

        Returns:
            np.ndarray: The softmax-transformed data.
        """
        x_max = np.max(data, axis=-1, keepdims=True)
        e_x = np.exp(data - x_max)
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def _parse_class_id_map(
        self, class_ids: Optional[Sequence[Union[str, int]]]
    ) -> Optional[dict]:
        """
        Parses a list of class IDs into a mapping from class index to class label.

        Args:
            class_ids (Optional[Sequence[Union[str, int]]]): A list of class labels.

        Returns:
            Optional[dict]: A dictionary mapping class indices to labels, or None if no class_ids are provided.
        """
        if class_ids is None:
            return None
        class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
        return class_id_map

    def __call__(
        self, preds: np.ndarray, topk: int = 5
    ) -> Tuple[np.ndarray, List[np.ndarray], List[List[str]]]:
        """
        Selects the top-k predictions from the classification output.

        Args:
            preds (np.ndarray): A 2D array of prediction scores.
            topk (int): The number of top predictions to return.

        Returns:
            Tuple[np.ndarray, List[np.ndarray], List[List[str]]]: A tuple containing:
                - An array of indices of the top-k predictions.
                - A list of arrays of scores for the top-k predictions.
                - A list of lists of label names for the top-k predictions.
        """
        preds[0] = self.softmax(preds[0])
        indexes = preds[0].argsort(axis=1)[:, -topk:][:, ::-1].astype("int32")
        scores = [
            list(np.around(pred[index], decimals=5))
            for pred, index in zip(preds[0], indexes)
        ]
        label_names = [[self.class_id_map[i] for i in index] for index in indexes]
        return indexes, scores, label_names


@benchmark.timeit
class ToBatch:
    """A class for batching videos."""

    def __call__(self, videos: List[np.ndarray]) -> List[np.ndarray]:
        """Call method to stack videos into a batch.

        Args:
            videos (list of np.ndarrays): List of videos to process.

        Returns:
            list of np.ndarrays: List containing a stacked tensor of the videos.
        """
        return [np.concatenate(videos, axis=0).astype(dtype=np.float32, copy=False)]
