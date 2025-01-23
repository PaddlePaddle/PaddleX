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

from typing import Any, Dict, Optional
from ..base import BasePipeline
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption


class PP_ChatOCR_Pipeline(BasePipeline):
    """PP-ChatOCR Pipeline"""

    def __init__(
        self,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the pp-chatocrv3-doc pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

    def visual_predict(self):
        """
        This function takes an input image or a list of images and performs various visual
        prediction tasks such as document orientation classification, document unwarping,
        general OCR, seal recognition, and table recognition based on the provided flags.
        """

        raise NotImplementedError(
            "The method `visual_predict` has not been implemented yet."
        )

    def save_visual_info_list(self):
        """
        Save the visual info list to the specified file path.
        """
        raise NotImplementedError(
            "The method `save_visual_info_list` has not been implemented yet."
        )

    def load_visual_info_list(self):
        """
        Loads visual info list from a file.
        """
        raise NotImplementedError(
            "The method `load_visual_info_list` has not been implemented yet."
        )

    def build_vector(self):
        """
        Build a vector representation from visual information.
        """
        raise NotImplementedError(
            "The method `build_vector` has not been implemented yet."
        )

    def save_vector(self):
        """
        Save the vector information to a specified path.
        """
        raise NotImplementedError(
            "The method `save_vector` has not been implemented yet."
        )

    def load_vector(self):
        """
        Loads vector information from a file.
        """
        raise NotImplementedError(
            "The method `load_vector` has not been implemented yet."
        )

    def chat(self):
        """
        Generates chat results based on the provided key list and visual information.
        """
        raise NotImplementedError("The method `chat` has not been implemented yet.")

    def predict(self, *args, **kwargs) -> None:
        logging.error(
            "PP-ChatOCR Pipeline do not support to call `predict()` directly! Please invoke `visual_predict`, `build_vector`, `chat` sequentially to obtain the result."
        )
        return
