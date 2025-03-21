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

from __future__ import division, print_function, unicode_literals

import copy
import json
from collections import defaultdict
from copy import deepcopy

import numpy as np


def order_by_tbyx(ocr_info):
    res = sorted(ocr_info, key=lambda r: (r["bbox"][1], r["bbox"][0]))
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            if abs(res[j + 1]["bbox"][1] - res[j]["bbox"][1]) < 20 and (
                res[j + 1]["bbox"][0] < res[j]["bbox"][0]
            ):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


class VQATokenLabelEncode(object):
    """
    Label encode for NLP VQA methods
    """

    def __init__(
        self,
        class_path,
        contains_re=False,
        add_special_ids=False,
        algorithm="LayoutXLM",
        use_textline_bbox_info=True,
        order_method=None,
        infer_mode=False,
        ocr_engine=None,
        **kwargs
    ):
        super(VQATokenLabelEncode, self).__init__()
        from paddlenlp.transformers import (
            LayoutLMTokenizer,
            LayoutLMv2Tokenizer,
            LayoutXLMTokenizer,
        )

        tokenizer_dict = {
            "LayoutXLM": {
                "class": LayoutXLMTokenizer,
                "pretrained_model": "layoutxlm-base-uncased",
            },
            "LayoutLM": {
                "class": LayoutLMTokenizer,
                "pretrained_model": "layoutlm-base-uncased",
            },
            "LayoutLMv2": {
                "class": LayoutLMv2Tokenizer,
                "pretrained_model": "layoutlmv2-base-uncased",
            },
        }
        self.contains_re = contains_re
        tokenizer_config = tokenizer_dict[algorithm]
        self.tokenizer = tokenizer_config["class"].from_pretrained(
            tokenizer_config["pretrained_model"]
        )
        self.label2id_map, id2label_map = load_vqa_bio_label_maps(class_path)
        self.add_special_ids = add_special_ids
        self.infer_mode = infer_mode
        self.ocr_engine = ocr_engine
        self.use_textline_bbox_info = use_textline_bbox_info
        self.order_method = order_method
        assert self.order_method in [None, "tb-yx"]

    def split_bbox(self, bbox, text, tokenizer):
        words = text.split()
        token_bboxes = []
        curr_word_idx = 0
        x1, y1, x2, y2 = bbox
        unit_w = (x2 - x1) / len(text)
        for idx, word in enumerate(words):
            curr_w = len(word) * unit_w
            word_bbox = [x1, y1, x1 + curr_w, y2]
            token_bboxes.extend([word_bbox] * len(tokenizer.tokenize(word)))
            x1 += (len(word) + 1) * unit_w
        return token_bboxes

    def filter_empty_contents(self, ocr_info):
        """
        find out the empty texts and remove the links
        """
        new_ocr_info = []
        empty_index = []
        for idx, info in enumerate(ocr_info):
            if len(info["transcription"]) > 0:
                new_ocr_info.append(copy.deepcopy(info))
            else:
                empty_index.append(info["id"])

        for idx, info in enumerate(new_ocr_info):
            new_link = []
            for link in info["linking"]:
                if link[0] in empty_index or link[1] in empty_index:
                    continue
                new_link.append(link)
            new_ocr_info[idx]["linking"] = new_link
        return new_ocr_info

    def __call__(self, data):
        # load bbox and label info
        ocr_info = self._load_ocr_info(data)

        for idx in range(len(ocr_info)):
            if "bbox" not in ocr_info[idx]:
                ocr_info[idx]["bbox"] = self.trans_poly_to_bbox(ocr_info[idx]["points"])

        if self.order_method == "tb-yx":
            ocr_info = order_by_tbyx(ocr_info)

        # for re
        train_re = self.contains_re and not self.infer_mode
        if train_re:
            ocr_info = self.filter_empty_contents(ocr_info)

        height, width, _ = data["image"].shape

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        segment_offset_id = []
        gt_label_list = []

        entities = []

        if train_re:
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()

        data["ocr_info"] = copy.deepcopy(ocr_info)

        for info in ocr_info:
            text = info["transcription"]
            if len(text) <= 0:
                continue
            if train_re:
                # for re
                if len(text) == 0:
                    empty_entity.add(info["id"])
                    continue
                id2label[info["id"]] = info["label"]
                relations.extend([tuple(sorted(l)) for l in info["linking"]])
            # smooth_box
            info["bbox"] = self.trans_poly_to_bbox(info["points"])

            encode_res = self.tokenizer.encode(
                text,
                pad_to_max_seq_len=False,
                return_attention_mask=True,
                return_token_type_ids=True,
            )

            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

            if self.use_textline_bbox_info:
                bbox = [info["bbox"]] * len(encode_res["input_ids"])
            else:
                bbox = self.split_bbox(
                    info["bbox"], info["transcription"], self.tokenizer
                )
            if len(bbox) <= 0:
                continue
            bbox = self._smooth_box(bbox, height, width)
            if self.add_special_ids:
                bbox.insert(0, [0, 0, 0, 0])
                bbox.append([0, 0, 0, 0])

            # parse label
            if not self.infer_mode:
                label = info["label"]
                gt_label = self._parse_label(label, encode_res)

            # construct entities for re
            if train_re:
                if gt_label[0] != self.label2id_map["O"]:
                    entity_id_to_index_map[info["id"]] = len(entities)
                    label = label.upper()
                    entities.append(
                        {
                            "start": len(input_ids_list),
                            "end": len(input_ids_list) + len(encode_res["input_ids"]),
                            "label": label.upper(),
                        }
                    )
            else:
                entities.append(
                    {
                        "start": len(input_ids_list),
                        "end": len(input_ids_list) + len(encode_res["input_ids"]),
                        "label": "O",
                    }
                )
            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend(bbox)
            words_list.append(text)
            segment_offset_id.append(len(input_ids_list))
            if not self.infer_mode:
                gt_label_list.extend(gt_label)

        data["input_ids"] = input_ids_list
        data["token_type_ids"] = token_type_ids_list
        data["bbox"] = bbox_list
        data["attention_mask"] = [1] * len(input_ids_list)
        data["labels"] = gt_label_list
        data["segment_offset_id"] = segment_offset_id
        data["tokenizer_params"] = dict(
            padding_side=self.tokenizer.padding_side,
            pad_token_type_id=self.tokenizer.pad_token_type_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        data["entities"] = entities

        if train_re:
            data["relations"] = relations
            data["id2label"] = id2label
            data["empty_entity"] = empty_entity
            data["entity_id_to_index_map"] = entity_id_to_index_map
        return data

    def trans_poly_to_bbox(self, poly):
        x1 = int(np.min([p[0] for p in poly]))
        x2 = int(np.max([p[0] for p in poly]))
        y1 = int(np.min([p[1] for p in poly]))
        y2 = int(np.max([p[1] for p in poly]))
        return [x1, y1, x2, y2]

    def _load_ocr_info(self, data):
        if self.infer_mode:
            ocr_result = self.ocr_engine.ocr(data["image"], cls=False)[0]
            ocr_info = []
            for res in ocr_result:
                ocr_info.append(
                    {
                        "transcription": res[1][0],
                        "bbox": self.trans_poly_to_bbox(res[0]),
                        "points": res[0],
                    }
                )
            return ocr_info
        else:
            info = data["label"]
            # read text info
            info_dict = json.loads(info)
            return info_dict

    def _smooth_box(self, bboxes, height, width):
        bboxes = np.array(bboxes)
        bboxes[:, 0] = bboxes[:, 0] * 1000 / width
        bboxes[:, 2] = bboxes[:, 2] * 1000 / width
        bboxes[:, 1] = bboxes[:, 1] * 1000 / height
        bboxes[:, 3] = bboxes[:, 3] * 1000 / height
        bboxes = bboxes.astype("int64").tolist()
        return bboxes

    def _parse_label(self, label, encode_res):
        gt_label = []
        if label.lower() in ["other", "others", "ignore"]:
            gt_label.extend([0] * len(encode_res["input_ids"]))
        else:
            gt_label.append(self.label2id_map[("b-" + label).upper()])
            gt_label.extend(
                [self.label2id_map[("i-" + label).upper()]]
                * (len(encode_res["input_ids"]) - 1)
            )
        return gt_label


class VQATokenPad(object):
    def __init__(
        self,
        max_seq_len=512,
        pad_to_max_seq_len=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation_strategy="longest_first",
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        infer_mode=False,
        **kwargs
    ):

        self.max_seq_len = max_seq_len
        self.pad_to_max_seq_len = max_seq_len
        self.return_attention_mask = return_attention_mask
        self.return_token_type_ids = return_token_type_ids
        self.truncation_strategy = truncation_strategy
        self.return_overflowing_tokens = return_overflowing_tokens
        self.return_special_tokens_mask = return_special_tokens_mask
        self.infer_mode = infer_mode

    def __call__(self, data):
        import paddle

        self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        needs_to_be_padded = (
            self.pad_to_max_seq_len and len(data["input_ids"]) < self.max_seq_len
        )

        if needs_to_be_padded:
            if "tokenizer_params" in data:
                tokenizer_params = data.pop("tokenizer_params")
            else:
                tokenizer_params = dict(
                    padding_side="right", pad_token_type_id=0, pad_token_id=1
                )

            difference = self.max_seq_len - len(data["input_ids"])
            if tokenizer_params["padding_side"] == "right":
                if self.return_attention_mask:
                    data["attention_mask"] = [1] * len(data["input_ids"]) + [
                        0
                    ] * difference
                if self.return_token_type_ids:
                    data["token_type_ids"] = (
                        data["token_type_ids"]
                        + [tokenizer_params["pad_token_type_id"]] * difference
                    )
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = (
                        data["special_tokens_mask"] + [1] * difference
                    )
                data["input_ids"] = (
                    data["input_ids"] + [tokenizer_params["pad_token_id"]] * difference
                )
                if not self.infer_mode:
                    data["labels"] = (
                        data["labels"] + [self.pad_token_label_id] * difference
                    )
                data["bbox"] = data["bbox"] + [[0, 0, 0, 0]] * difference
            elif tokenizer_params["padding_side"] == "left":
                if self.return_attention_mask:
                    data["attention_mask"] = [0] * difference + [1] * len(
                        data["input_ids"]
                    )
                if self.return_token_type_ids:
                    data["token_type_ids"] = [
                        tokenizer_params["pad_token_type_id"]
                    ] * difference + data["token_type_ids"]
                if self.return_special_tokens_mask:
                    data["special_tokens_mask"] = [1] * difference + data[
                        "special_tokens_mask"
                    ]
                data["input_ids"] = [
                    tokenizer_params["pad_token_id"]
                ] * difference + data["input_ids"]
                if not self.infer_mode:
                    data["labels"] = [self.pad_token_label_id] * difference + data[
                        "labels"
                    ]
                data["bbox"] = [[0, 0, 0, 0]] * difference + data["bbox"]
        else:
            if self.return_attention_mask:
                data["attention_mask"] = [1] * len(data["input_ids"])

        for key in data:
            if key in [
                "input_ids",
                "labels",
                "token_type_ids",
                "bbox",
                "attention_mask",
            ]:
                if self.infer_mode:
                    if key != "labels":
                        length = min(len(data[key]), self.max_seq_len)
                        data[key] = data[key][:length]
                    else:
                        continue
                data[key] = np.array(data[key], dtype="int64")
        return data


class VQASerTokenChunk(object):
    def __init__(self, max_seq_len=512, infer_mode=False, **kwargs):
        self.max_seq_len = max_seq_len
        self.infer_mode = infer_mode

    def __call__(self, data):
        encoded_inputs_all = []
        seq_len = len(data["input_ids"])
        for index in range(0, seq_len, self.max_seq_len):
            chunk_beg = index
            chunk_end = min(index + self.max_seq_len, seq_len)
            encoded_inputs_example = {}
            for key in data:
                if key in [
                    "label",
                    "input_ids",
                    "labels",
                    "token_type_ids",
                    "bbox",
                    "attention_mask",
                ]:
                    if self.infer_mode and key == "labels":
                        encoded_inputs_example[key] = data[key]
                    else:
                        encoded_inputs_example[key] = data[key][chunk_beg:chunk_end]
                else:
                    encoded_inputs_example[key] = data[key]

            encoded_inputs_all.append(encoded_inputs_example)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]


class VQAReTokenChunk(object):
    def __init__(
        self, max_seq_len=512, entities_labels=None, infer_mode=False, **kwargs
    ):
        self.max_seq_len = max_seq_len
        self.entities_labels = (
            {"HEADER": 0, "QUESTION": 1, "ANSWER": 2}
            if entities_labels is None
            else entities_labels
        )
        self.infer_mode = infer_mode

    def __call__(self, data):
        # prepare data
        entities = data.pop("entities")
        relations = data.pop("relations")
        encoded_inputs_all = []
        for index in range(0, len(data["input_ids"]), self.max_seq_len):
            item = {}
            for key in data:
                if key in [
                    "label",
                    "input_ids",
                    "labels",
                    "token_type_ids",
                    "bbox",
                    "attention_mask",
                ]:
                    if self.infer_mode and key == "labels":
                        item[key] = data[key]
                    else:
                        item[key] = data[key][index : index + self.max_seq_len]
                else:
                    item[key] = data[key]
            # select entity in current chunk
            entities_in_this_span = []
            global_to_local_map = {}  #
            for entity_id, entity in enumerate(entities):
                if (
                    index <= entity["start"] < index + self.max_seq_len
                    and index <= entity["end"] < index + self.max_seq_len
                ):
                    entity["start"] = entity["start"] - index
                    entity["end"] = entity["end"] - index
                    global_to_local_map[entity_id] = len(entities_in_this_span)
                    entities_in_this_span.append(entity)

            # select relations in current chunk
            relations_in_this_span = []
            for relation in relations:
                if (
                    index <= relation["start_index"] < index + self.max_seq_len
                    and index <= relation["end_index"] < index + self.max_seq_len
                ):
                    relations_in_this_span.append(
                        {
                            "head": global_to_local_map[relation["head"]],
                            "tail": global_to_local_map[relation["tail"]],
                            "start_index": relation["start_index"] - index,
                            "end_index": relation["end_index"] - index,
                        }
                    )
            item.update(
                {
                    "entities": self.reformat(entities_in_this_span),
                    "relations": self.reformat(relations_in_this_span),
                }
            )
            if len(item["entities"]) > 0:
                item["entities"]["label"] = [
                    self.entities_labels[x] for x in item["entities"]["label"]
                ]
                encoded_inputs_all.append(item)
        if len(encoded_inputs_all) == 0:
            return None
        return encoded_inputs_all[0]

    def reformat(self, data):
        new_data = defaultdict(list)
        for item in data:
            for k, v in item.items():
                new_data[k].append(v)
        return new_data


class VQASerTokenLayoutLMPostProcess(object):
    """Convert between text-label and text-index"""

    def __init__(self, class_path, **kwargs):
        super(VQASerTokenLayoutLMPostProcess, self).__init__()
        label2id_map, self.id2label_map = load_vqa_bio_label_maps(class_path)

        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

        self.id2label_map_for_show = dict()
        for key in self.label2id_map_for_draw:
            val = self.label2id_map_for_draw[key]
            if key == "O":
                self.id2label_map_for_show[val] = key
            if key.startswith("B-") or key.startswith("I-"):
                self.id2label_map_for_show[val] = key[2:]
            else:
                self.id2label_map_for_show[val] = key

    def __call__(self, preds, batch=None, *args, **kwargs):
        import paddle

        if isinstance(preds, tuple):
            preds = preds[0]
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()

        if batch is not None:
            return self._metric(preds, batch[5])
        else:
            return self._infer(preds, **kwargs)

    def _metric(self, preds, label):
        pred_idxs = preds.argmax(axis=2)
        decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        label_decode_out_list = [[] for _ in range(pred_idxs.shape[0])]

        for i in range(pred_idxs.shape[0]):
            for j in range(pred_idxs.shape[1]):
                if label[i, j] != -100:
                    label_decode_out_list[i].append(self.id2label_map[label[i, j]])
                    decode_out_list[i].append(self.id2label_map[pred_idxs[i, j]])
        return decode_out_list, label_decode_out_list

    def _infer(self, preds, segment_offset_ids, ocr_infos):
        results = []

        for pred, segment_offset_id, ocr_info in zip(
            preds, segment_offset_ids, ocr_infos
        ):
            pred = np.argmax(pred, axis=1)
            pred = [self.id2label_map[idx] for idx in pred]

            for idx in range(len(segment_offset_id)):
                if idx == 0:
                    start_id = 0
                else:
                    start_id = segment_offset_id[idx - 1]

                end_id = segment_offset_id[idx]

                curr_pred = pred[start_id:end_id]
                curr_pred = [self.label2id_map_for_draw[p] for p in curr_pred]

                if len(curr_pred) <= 0:
                    pred_id = 0
                else:
                    counts = np.bincount(curr_pred)
                    pred_id = np.argmax(counts)
                ocr_info[idx]["pred_id"] = int(pred_id)
                ocr_info[idx]["pred"] = self.id2label_map_for_show[int(pred_id)]
            results.append(ocr_info)
        return results
