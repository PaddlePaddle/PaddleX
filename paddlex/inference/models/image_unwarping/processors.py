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

from typing import List, Tuple, Union

import numpy as np

from ...utils.benchmark import benchmark


@benchmark.timeit
class DocTrPostProcess:
    """
    Post-processing class for cropping regions from images (though currently only performs scaling and color channel adjustments).

    Attributes:
        scale (np.float32): A scaling factor to be applied to the image pixel values.
            Defaults to 255.0 if not provided.

    Methods:
        __call__(imgs: List[Union[np.ndarray, Tuple[np.ndarray, ...]]]) -> List[np.ndarray]:
            Call method to process a list of images.
        doctr(pred: Union[np.ndarray, Tuple[np.ndarray, ...]]) -> np.ndarray:
            Method to process a single image or a tuple/list containing an image.
    """

    def __init__(self, scale: Union[str, float, None] = None):
        """
        Initializes the DocTrPostProcess class with a scaling factor.

        Args:
            scale (Union[str, float, None]): A scaling factor for the image pixel values.
                If a string is provided, it will be converted to a float. Defaults to 255.0.
        """
        super().__init__()
        self.scale = (
            np.float32(scale) if isinstance(scale, (str, float)) else np.float32(255.0)
        )

    def __call__(
        self, imgs: List[Union[np.ndarray, Tuple[np.ndarray, ...]]]
    ) -> List[np.ndarray]:
        """
        Processes a list of images using the `doctr` method.

        Args:
            imgs (List[Union[np.ndarray, Tuple[np.ndarray, ...]]]): A list of images to process.
                Each image can be a numpy array or a tuple containing a numpy array.

        Returns:
            List[np.ndarray]: A list of processed images.
        """
        return [self.doctr(img) for img in imgs]

    def doctr(self, pred: Union[np.ndarray, Tuple[np.ndarray, ...]]) -> np.ndarray:
        """
        Processes a single image.

        Args:
            pred (Union[np.ndarray, Tuple[np.ndarray, ...]]): The image to process, which can be
                a numpy array or a tuple containing a numpy array. Only the first element is used if it's a tuple.

        Returns:
            np.ndarray: The processed image.

        Raises:
            AssertionError: If the input is not a numpy array.
        """
        if isinstance(pred, tuple):
            im = pred[0]
        else:
            im = pred
        assert isinstance(
            im, np.ndarray
        ), "Invalid input 'im' in DocTrPostProcess. Expected a numpy array."
        im = im.squeeze()
        im = im.transpose(1, 2, 0)
        im *= self.scale
        im = im[:, :, ::-1]
        im = im.astype("uint8", copy=False)
        return im
