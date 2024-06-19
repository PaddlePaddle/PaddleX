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



from ...modules.base import create_model
from ...modules.text_detection.predictor import transforms as text_det_T
from .utils import draw_ocr_box_txt


class OCRPipeline(object):
    """OCR Pipeline
    """

    def __init__(self,
                 text_det_model_name,
                 text_rec_model_name,
                 text_det_model_dir=None,
                 text_rec_model_dir=None,
                 text_det_kernel_option=None,
                 text_rec_kernel_option=None,
                 output_dir="output"):
        self.output_dir = output_dir
        text_det_kernel_option = self.get_kernel_option(
        ) if text_det_kernel_option is None else text_det_kernel_option
        text_rec_kernel_option = self.get_kernel_option(
        ) if text_rec_kernel_option is None else text_rec_kernel_option

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
            text_det_model_name,
            text_det_model_dir,
            kernel_option=text_det_kernel_option,
            post_transforms=text_det_post_transforms)
        self.text_rec_model = create_model(
            text_rec_model_name,
            text_rec_model_dir,
            kernel_option=text_rec_kernel_option)

    def __call__(self, input_path):
        result = self.text_det_model.predict({"input_path": input_path})
        all_rec_result = []
        for i, img in enumerate(result["sub_imgs"]):
            rec_result = self.text_rec_model.predict({"image": img})
            all_rec_result.append(rec_result["rec_text"][0])
        result["rec_text"] = all_rec_result
        return result
