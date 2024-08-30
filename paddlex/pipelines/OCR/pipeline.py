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
import cv2

from ..base import BasePipeline
from ...modules.text_detection.model_list import MODELS as text_det_models
from ...modules.text_recognition.model_list import MODELS as text_rec_models
from ...modules import create_model, PaddleInferenceOption
from ...modules.text_detection import transforms as text_det_T
from .utils import draw_ocr_box_txt


class OCRPipeline(BasePipeline):
    """OCR Pipeline
    """
    entities = "OCR"

    def __init__(
            self,
            text_det_model_name=None,
            text_rec_model_name=None,
            text_det_model_dir=None,
            text_rec_model_dir=None,
            text_det_kernel_option=None,
            text_rec_kernel_option=None,
            output="./",
            device="gpu",
            **kwargs, ):
        super().__init__(**kwargs)
        self.text_det_model_name = text_det_model_name
        self.text_rec_model_name = text_rec_model_name
        self.text_det_model_dir = text_det_model_dir
        self.text_rec_model_dir = text_rec_model_dir
        self.output = output
        self.device = device
        self.text_det_kernel_option = text_det_kernel_option
        self.text_rec_kernel_option = text_rec_kernel_option
        if self.text_det_model_name is not None and self.text_rec_model_name is not None:
            self.load_model()

    def check_model_name(self):
        """ check that model name is valid
        """
        assert self.text_det_model_name in text_det_models, f"The model name({self.text_det_model_name}) error. \
Only support: {text_det_models}."

        assert self.text_rec_model_name in text_rec_models, f"The model name({self.text_rec_model_name}) error. \
Only support: {text_rec_models}."

    def load_model(self):
        """load model predictor
        """
        self.check_model_name()
        text_det_kernel_option = self.get_kernel_option(
        ) if self.text_det_kernel_option is None else self.text_det_kernel_option
        text_rec_kernel_option = self.get_kernel_option(
        ) if self.text_rec_kernel_option is None else self.text_rec_kernel_option
        text_det_post_transforms = [
            text_det_T.DBPostProcess(
                thresh=0.3,
                box_thresh=0.6,
                max_candidates=1000,
                unclip_ratio=1.5,
                use_dilation=False,
                score_mode='fast',
                box_type='quad'),
            # TODO
            text_det_T.CropByPolys(det_box_type="foo")
        ]

        self.text_det_model = create_model(
            self.text_det_model_name,
            self.text_det_model_dir,
            kernel_option=text_det_kernel_option,
            post_transforms=text_det_post_transforms)
        self.text_rec_model = create_model(
            self.text_rec_model_name,
            self.text_rec_model_dir,
            kernel_option=text_rec_kernel_option,
            disable_print=self.disable_print,
            disable_save=self.disable_save, )

    def predict(self, input):
        """predict
        """
        result = self.text_det_model.predict(input)
        all_rec_result = []
        for i, img in enumerate(result["sub_imgs"]):
            rec_result = self.text_rec_model.predict({"image": img})
            all_rec_result.append(rec_result["rec_text"][0])
        result["rec_text"] = all_rec_result

        if self.output is not None:
            draw_img = draw_ocr_box_txt(result['original_image'],
                                        result['dt_polys'], result["rec_text"])
            fn = os.path.basename(result['input_path'])
            cv2.imwrite(
                os.path.join(self.output, fn),
                draw_img[:, :, ::-1], )

        return result

    def update_model(self, model_name_list, model_dir_list):
        """update model

        Args:
            model_name_list (list): list of model name.
            model_dir_list (list): list of model directory.
        """
        assert len(model_name_list) == 2
        self.text_det_model_name = model_name_list[0]
        self.text_rec_model_name = model_name_list[1]
        if model_dir_list:
            assert len(model_dir_list) == 2
            self.text_det_model_dir = model_dir_list[0]
            self.text_rec_model_dir = model_dir_list[1]

    def get_kernel_option(self):
        """get kernel option
        """
        kernel_option = PaddleInferenceOption()
        kernel_option.set_device(self.device)
        return kernel_option

    def get_input_keys(self):
        """get dict keys of input argument input
        """
        return self.text_det_model.get_input_keys()
