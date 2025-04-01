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

import numpy as np

from ....utils.deps import pipeline_requires_extra
from ..pp_shitu_v2 import ShiTuV2Pipeline
from .result import FaceRecResult


@pipeline_requires_extra("cv")
class FaceRecPipeline(ShiTuV2Pipeline):
    """Face Recognition Pipeline"""

    entities = "face_recognition"

    def get_rec_result(
        self, raw_img, det_res, indexer, rec_threshold, hamming_radius, topk
    ):
        if len(det_res["boxes"]) == 0:
            return {"label": [], "score": []}
        subs_of_img = list(self.crop_by_boxes(raw_img, det_res["boxes"]))
        img_list = [img["img"] for img in subs_of_img]
        all_rec_res = list(self.rec_model(img_list))
        all_rec_res = indexer(
            [rec_res["feature"] for rec_res in all_rec_res],
            score_thres=rec_threshold,
            hamming_radius=hamming_radius,
            topk=topk,
        )
        output = {"label": [], "score": []}
        for res in all_rec_res:
            output["label"].append(res["label"])
            output["score"].append(res["score"])
        return output

    def get_final_result(self, input_data, raw_img, det_res, rec_res):
        single_img_res = {"input_path": input_data, "input_img": raw_img, "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            rec_scores = rec_res["score"][i]
            if isinstance(rec_scores, np.ndarray):
                rec_scores = rec_scores.tolist()
            labels = rec_res["label"][i]
            single_img_res["boxes"].append(
                {
                    "labels": labels,
                    "rec_scores": rec_scores,
                    "det_score": obj["score"],
                    "coordinate": obj["coordinate"],
                }
            )
        return FaceRecResult(single_img_res)
