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

import pickle
from pathlib import Path

import numpy as np

from ....utils import logging
from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.io import YAMLReader, YAMLWriter

if is_dep_available("faiss-cpu"):
    import faiss


@class_requires_deps("faiss-cpu")
class IndexData:
    VECTOR_FN = "vector"
    VECTOR_SUFFIX = ".index"
    IDMAP_FN = "id_map"
    IDMAP_SUFFIX = ".yaml"

    def __init__(self, index, index_info):
        self._index = index
        self._index_info = index_info
        self._id_map = index_info["id_map"]
        self._metric_type = index_info["metric_type"]
        self._index_type = index_info["index_type"]

    @property
    def index(self):
        return self._index

    @property
    def index_bytes(self):
        return faiss.serialize_index(self._index)

    @property
    def id_map(self):
        return self._id_map

    @property
    def metric_type(self):
        return self._metric_type

    @property
    def index_type(self):
        return self._index_type

    @property
    def index_info(self):
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "id_map": self._convert_int(self.id_map),
        }

    @classmethod
    def from_bytes(cls, bytes):
        tup = pickle.loads(bytes)
        index = faiss.deserialize_index(tup[0])
        return cls(index, tup[1])

    def to_bytes(self):
        tup = (faiss.serialize_index(self._index), self.index_info)
        return pickle.dumps(tup)

    def _convert_int(self, id_map):
        return {int(k): str(v) for k, v in id_map.items()}

    @staticmethod
    def _convert_int64(id_map):
        return {np.int64(k): str(v) for k, v in id_map.items()}

    def save(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        vector_path = (save_dir / f"{self.VECTOR_FN}{self.VECTOR_SUFFIX}").as_posix()
        index_info_path = (save_dir / f"{self.IDMAP_FN}{self.IDMAP_SUFFIX}").as_posix()

        if self.metric_type in FaissBuilder.BINARY_METRIC_TYPE:
            faiss.write_index_binary(self.index, vector_path)
        else:
            faiss.write_index(self.index, vector_path)

        yaml_writer = YAMLWriter()
        yaml_writer.write(
            index_info_path,
            self.index_info,
            default_flow_style=False,
            allow_unicode=True,
        )

    @classmethod
    def load(cls, index):
        if isinstance(index, str):
            index_root = Path(index)
            vector_path = index_root / f"{cls.VECTOR_FN}{cls.VECTOR_SUFFIX}"
            index_info_path = index_root / f"{cls.IDMAP_FN}{cls.IDMAP_SUFFIX}"

            assert (
                vector_path.exists()
            ), f"Not found the {cls.VECTOR_FN}{cls.VECTOR_SUFFIX} file in {index}!"
            assert (
                index_info_path.exists()
            ), f"Not found the {cls.IDMAP_FN}{cls.IDMAP_SUFFIX} file in {index}!"

            yaml_reader = YAMLReader()
            index_info = yaml_reader.read(index_info_path)
            assert (
                "id_map" in index_info
                and "metric_type" in index_info
                and "index_type" in index_info
            ), f"The index_info file({index_info_path}) may have been damaged, `id_map` or `metric_type` or `index_type` not found in `index_info`."
            id_map = IndexData._convert_int64(index_info["id_map"])

            if index_info["metric_type"] in FaissBuilder.BINARY_METRIC_TYPE:
                index = faiss.read_index_binary(vector_path.as_posix())
            else:
                index = faiss.read_index(vector_path.as_posix())
            assert index.ntotal == len(
                id_map
            ), "data number in index is not equal in in id_map"

            return index, id_map, index_info["metric_type"], index_info["index_type"]
        else:
            assert isinstance(index, IndexData)
            return index.index, index.id_map, index.metric_type, index.index_type


class FaissIndexer:

    def __init__(
        self,
        index,
    ):
        super().__init__()
        self._indexer, self.id_map, self.metric_type, index_type = IndexData.load(index)

    def __call__(self, feature, score_thres, hamming_radius, topk):
        scores_list, ids_list = self._indexer.search(np.array(feature), topk)
        preds = []
        for scores, ids in zip(scores_list, ids_list):
            preds.append({"score": [], "label": []})
            for score, id in zip(scores, ids):
                if id >= 0:
                    preds[-1]["score"].append(score)
                    preds[-1]["label"].append(self.id_map[id])

        if self.metric_type in FaissBuilder.BINARY_METRIC_TYPE:
            idxs = np.where(scores_list[:, 0] > hamming_radius)[0]
        else:
            idxs = np.where(scores_list[:, 0] < score_thres)[0]
        for idx in idxs:
            preds[idx] = {"score": None, "label": None}
        return preds


@class_requires_deps("faiss-cpu")
class FaissBuilder:

    SUPPORT_METRIC_TYPE = ("hamming", "IP", "L2")
    SUPPORT_INDEX_TYPE = ("Flat", "IVF", "HNSW32")
    BINARY_METRIC_TYPE = ("hamming",)
    BINARY_SUPPORT_INDEX_TYPE = ("Flat", "IVF", "BinaryHash")

    @classmethod
    def _get_index_type(cls, metric_type, index_type, num=None):
        # if IVF method, cal ivf number automatically
        if index_type == "IVF":
            index_type = index_type + str(min(int(num // 8), 65536))
            if metric_type in cls.BINARY_METRIC_TYPE:
                index_type += ",BFlat"
            else:
                index_type += ",Flat"

        # for binary index, add B at head of index_type
        if metric_type in cls.BINARY_METRIC_TYPE:
            assert (
                index_type in cls.BINARY_SUPPORT_INDEX_TYPE
            ), f"The metric type({metric_type}) only support {cls.BINARY_SUPPORT_INDEX_TYPE} index types!"
            index_type = "B" + index_type

        if index_type == "HNSW32":
            logging.warning("The HNSW32 method dose not support 'remove' operation")
            index_type = "HNSW32"

        if index_type == "Flat":
            index_type = "Flat"

        return index_type

    @classmethod
    def _get_metric_type(cls, metric_type):
        if metric_type == "hamming":
            return faiss.METRIC_Hamming
        elif metric_type == "jaccard":
            return faiss.METRIC_Jaccard
        elif metric_type == "IP":
            return faiss.METRIC_INNER_PRODUCT
        elif metric_type == "L2":
            return faiss.METRIC_L2

    @classmethod
    def build(
        cls,
        gallery_imgs,
        gallery_label,
        predict_func,
        metric_type="IP",
        index_type="HNSW32",
    ):
        assert (
            index_type in cls.SUPPORT_INDEX_TYPE
        ), f"Supported index types only: {cls.SUPPORT_INDEX_TYPE}!"

        assert (
            metric_type in cls.SUPPORT_METRIC_TYPE
        ), f"Supported metric types only: {cls.SUPPORT_METRIC_TYPE}!"

        if isinstance(gallery_label, str):
            gallery_docs, gallery_list = cls.load_gallery(gallery_label, gallery_imgs)
        else:
            gallery_docs, gallery_list = gallery_label, gallery_imgs

        features = [res["feature"] for res in predict_func(gallery_list)]
        dtype = np.uint8 if metric_type in cls.BINARY_METRIC_TYPE else np.float32
        features = np.array(features).astype(dtype)
        vector_num, vector_dim = features.shape

        if metric_type in cls.BINARY_METRIC_TYPE:
            index = faiss.index_binary_factory(
                vector_dim,
                cls._get_index_type(metric_type, index_type, vector_num),
                cls._get_metric_type(metric_type),
            )
        else:
            index = faiss.index_factory(
                vector_dim,
                cls._get_index_type(metric_type, index_type, vector_num),
                cls._get_metric_type(metric_type),
            )
            index = faiss.IndexIDMap2(index)
        ids = {}

        # calculate id for new data
        index, ids = cls._add_gallery(
            metric_type, index, ids, features, gallery_docs, mode="new"
        )
        return IndexData(
            index, {"id_map": ids, "metric_type": metric_type, "index_type": index_type}
        )

    @classmethod
    def remove(
        cls,
        remove_ids,
        index,
    ):
        index, ids, metric_type, index_type = IndexData.load(index)
        if index_type == "HNSW32":
            raise RuntimeError(
                "The index_type: HNSW32 dose not support 'remove' operation"
            )
        if isinstance(remove_ids, str):
            lines = []
            with open(remove_ids) as f:
                lines = f.readlines()
            remove_ids = []
            for line in lines:
                id_ = int(line.strip().split(" ")[0])
                remove_ids.append(id_)
            remove_ids = np.asarray(remove_ids)
        else:
            remove_ids = np.asarray(remove_ids)

        # remove ids in id_map, remove index data in faiss index
        index.remove_ids(remove_ids)
        ids = {k: v for k, v in ids.items() if k not in remove_ids}
        return IndexData(
            index, {"id_map": ids, "metric_type": metric_type, "index_type": index_type}
        )

    @classmethod
    def append(cls, gallery_imgs, gallery_label, predict_func, index):
        index, ids, metric_type, index_type = IndexData.load(index)
        assert (
            metric_type in cls.SUPPORT_METRIC_TYPE
        ), f"Supported metric types only: {cls.SUPPORT_METRIC_TYPE}!"

        if isinstance(gallery_label, str):
            gallery_docs, gallery_list = cls.load_gallery(gallery_label, gallery_imgs)
        else:
            gallery_docs, gallery_list = gallery_label, gallery_imgs

        features = [res["feature"] for res in predict_func(gallery_list)]
        dtype = np.uint8 if metric_type in cls.BINARY_METRIC_TYPE else np.float32
        features = np.array(features).astype(dtype)

        # calculate id for new data
        index, ids = cls._add_gallery(
            metric_type, index, ids, features, gallery_docs, mode="append"
        )
        return IndexData(
            index, {"id_map": ids, "metric_type": metric_type, "index_type": index_type}
        )

    @classmethod
    def _add_gallery(
        cls, metric_type, index, ids, gallery_features, gallery_docs, mode
    ):
        start_id = max(ids.keys()) + 1 if ids else 0
        ids_now = (np.arange(0, len(gallery_docs)) + start_id).astype(np.int64)

        # only train when new index file
        if mode == "new":
            if metric_type in cls.BINARY_METRIC_TYPE:
                index.add(gallery_features)
            else:
                index.train(gallery_features)

        if metric_type not in cls.BINARY_METRIC_TYPE:
            index.add_with_ids(gallery_features, ids_now)
        # TODO(gaotingquan): how append when using hamming metric type
        # else:
        #   pass

        for i, d in zip(list(ids_now), gallery_docs):
            ids[i] = d
        return index, ids

    @classmethod
    def load_gallery(cls, gallery_label_path, gallery_imgs_root="", delimiter=" "):
        lines = []
        files = []
        labels = []
        root = Path(gallery_imgs_root)
        with open(gallery_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            path, label = line.strip().split(delimiter)
            file_path = root / path
            files.append(file_path.as_posix())
            labels.append(label)
        return labels, files
