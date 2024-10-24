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

import numpy as np
from ...components import CropByBoxes, FaissIndexer
from ...components.retrieval.faiss import FaissBuilder
from ...results import ShiTuResult
from ..base import BasePipeline
from ...utils.io import ImageReader


class ShiTuPipeline(BasePipeline):
    """ShiTu Pipeline"""

    entities = "PP-ShiTu"

    def __init__(
        self,
        det_model,
        rec_model,
        det_batch_size=1,
        rec_batch_size=1,
        index_dir=None,
        metric_type="IP",
        score_thres=None,
        hamming_radius=None,
        return_k=5,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(device, predictor_kwargs)
        self._build_predictor(det_model, rec_model)
        self.set_predictor(det_batch_size, rec_batch_size, device)
        self._indexer = (
            FaissIndexer(index_dir, metric_type, return_k, score_thres, hamming_radius)
            if index_dir
            else None
        )

    def _build_predictor(self, det_model, rec_model):
        self.det_model = self._create(model=det_model)
        self.rec_model = self._create(model=rec_model)
        self._crop_by_boxes = CropByBoxes()
        self._img_reader = ImageReader(backend="opencv")

    def set_predictor(self, det_batch_size=None, rec_batch_size=None, device=None):
        if det_batch_size:
            self.det_model.set_predictor(batch_size=det_batch_size)
        if rec_batch_size:
            self.rec_model.set_predictor(batch_size=rec_batch_size)
        if device:
            self.det_model.set_predictor(device=device)
            self.rec_model.set_predictor(device=device)

    def predict(self, input, **kwargs):
        assert self._indexer
        self.set_predictor(**kwargs)
        for det_res in self.det_model(input):
            rec_res = self.get_rec_result(det_res)
            yield self.get_final_result(det_res, rec_res)

    def get_rec_result(self, det_res):
        full_img = self._img_reader.read(det_res["input_path"])
        w, h = full_img.shape[:2]
        det_res["boxes"].append(
            {"cls_id": 0, "label": "full_img", "score": 0, "coordinate": [0, 0, h, w]}
        )
        subs_of_img = list(self._crop_by_boxes(det_res))
        img_list = [img["img"] for img in subs_of_img]
        all_rec_res = list(self.rec_model(img_list))
        all_rec_res = next(self._indexer(all_rec_res))
        output = {"label": [], "score": []}
        for res in all_rec_res:
            output["label"].append(res["label"])
            output["score"].append(res["score"])
        return output

    def get_final_result(self, det_res, rec_res):
        single_img_res = {"input_path": det_res["input_path"], "boxes": []}
        for i, obj in enumerate(det_res["boxes"]):
            rec_scores = rec_res["score"][i]
            labels = rec_res["label"][i]
            single_img_res["boxes"].append(
                {
                    "labels": labels,
                    "rec_scores": rec_scores,
                    "det_score": obj["score"],
                    "coordinate": obj["coordinate"],
                }
            )
        return ShiTuResult(single_img_res)

    def build_index(
        self,
        label_file,
        image_root,
        index_dir,
        mode="new",
        metric_type="IP",
        index_type="HNSW32",
        **kwargs,
    ):
        self.set_predictor(**kwargs)
        builder = FaissBuilder(
            self.rec_model.predict,
            mode=mode,
            metric_type=metric_type,
            index_type=index_type,
        )
        builder.build(label_file, image_root, index_dir)
        return
