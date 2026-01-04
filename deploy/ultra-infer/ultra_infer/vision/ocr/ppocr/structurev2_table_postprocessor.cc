// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ultra_infer/vision/ocr/ppocr/structurev2_table_postprocessor.h"

#include "ultra_infer/utils/perf.h"
#include "ultra_infer/vision/ocr/ppocr/utils/ocr_utils.h"

namespace ultra_infer {
namespace vision {
namespace ocr {

StructureV2TablePostprocessor::StructureV2TablePostprocessor() {
  initialized_ = false;
}

StructureV2TablePostprocessor::StructureV2TablePostprocessor(
    const std::string &dict_path, const std::string &box_shape)
    : box_shape(box_shape) {
  std::ifstream in(dict_path);

  FDASSERT(in, "Cannot open file %s to read.", dict_path.c_str());
  std::string line;
  dict_character.clear();
  dict_character.push_back("sos"); // add special character
  while (getline(in, line)) {
    dict_character.push_back(line);
  }

  if (merge_no_span_structure) {
    if (std::find(dict_character.begin(), dict_character.end(), "<td></td>") ==
        dict_character.end()) {
      dict_character.push_back("<td></td>");
    }
    for (auto it = dict_character.begin(); it != dict_character.end();) {
      if (*it == "<td>") {
        it = dict_character.erase(it);
      } else {
        ++it;
      }
    }
  }

  dict_character.push_back("eos"); // add special character
  dict.clear();
  for (size_t i = 0; i < dict_character.size(); i++) {
    dict[dict_character[i]] = int(i);
    if (dict_character[i] == "beg") {
      ignore_beg_token_idx = i;
    } else if (dict_character[i] == "end") {
      ignore_end_token_idx = i;
    }
  }
  dict_end_idx = dict_character.size() - 1;

  initialized_ = true;
}

bool StructureV2TablePostprocessor::SingleBatchPostprocessor(
    const float *structure_probs, const float *bbox_preds, size_t slice_dim,
    size_t prob_dim, size_t box_dim, int img_width, int img_height,
    float ratio_h, float ratio_w, int pad_h, int pad_w,
    std::vector<std::array<int, 8>> *boxes_result,
    std::vector<std::string> *structure_list_result) {
  structure_list_result->push_back("<html>");
  structure_list_result->push_back("<body>");
  structure_list_result->push_back("<table>");

  for (int i = 0; i < slice_dim; i++) {
    int structure_idx = 0;
    float structure_prob = structure_probs[i * prob_dim];
    for (int j = 0; j < prob_dim; j++) {
      if (structure_probs[i * prob_dim + j] > structure_prob) {
        structure_prob = structure_probs[i * prob_dim + j];
        structure_idx = j;
      }
    }

    if (structure_idx > 0 && structure_idx == dict_end_idx)
      break;

    if (structure_idx == ignore_end_token_idx ||
        structure_idx == ignore_beg_token_idx)
      continue;

    std::string text = dict_character[structure_idx];
    if (std::find(td_tokens.begin(), td_tokens.end(), text) !=
        td_tokens.end()) {
      std::array<int, 8> bbox;
      // box dim: en->4, ch->8

      if (box_dim == 4) {
        bbox[0] = bbox_preds[i * box_dim] * img_width;
        bbox[1] = bbox_preds[i * box_dim + 1] * img_height;

        bbox[2] = bbox_preds[i * box_dim + 2] * img_width;
        bbox[3] = bbox_preds[i * box_dim + 1] * img_height;

        bbox[4] = bbox_preds[i * box_dim + 2] * img_width;
        bbox[5] = bbox_preds[i * box_dim + 3] * img_height;

        bbox[6] = bbox_preds[i * box_dim] * img_width;
        bbox[7] = bbox_preds[i * box_dim + 3] * img_height;
      } else {
        for (int k = 0; k < 8; k++) {
          float bbox_pred = bbox_preds[i * box_dim + k];
          if (box_shape == "pad") {
            bbox[k] = int(k % 2 == 0 ? bbox_pred * pad_w / ratio_w
                                     : bbox_pred * pad_h / ratio_h);
          } else {
            bbox[k] = int(k % 2 == 0 ? bbox_pred * img_width
                                     : bbox_pred * img_height);
          }
        }
      }

      boxes_result->push_back(bbox);
    }
    structure_list_result->push_back(text);
  }
  structure_list_result->push_back("</table>");
  structure_list_result->push_back("</body>");
  structure_list_result->push_back("</html>");

  return true;
}

bool StructureV2TablePostprocessor::Run(
    const std::vector<FDTensor> &tensors,
    std::vector<std::vector<std::array<int, 8>>> *bbox_batch_list,
    std::vector<std::vector<std::string>> *structure_batch_list,
    const std::vector<std::array<float, 6>> &batch_det_img_info) {
  // Table have 2 output tensors.
  const FDTensor &structure_probs = tensors[1];
  const FDTensor &bbox_preds = tensors[0];

  const float *structure_probs_data =
      reinterpret_cast<const float *>(structure_probs.Data());
  size_t structure_probs_length =
      accumulate(structure_probs.shape.begin() + 1, structure_probs.shape.end(),
                 1, std::multiplies<int>());

  const float *bbox_preds_data =
      reinterpret_cast<const float *>(bbox_preds.Data());
  size_t bbox_preds_length =
      accumulate(bbox_preds.shape.begin() + 1, bbox_preds.shape.end(), 1,
                 std::multiplies<int>());
  size_t batch = bbox_preds.shape[0];
  size_t slice_dim = bbox_preds.shape[1];
  size_t prob_dim = structure_probs.shape[2];
  size_t box_dim = bbox_preds.shape[2];

  bbox_batch_list->resize(batch);
  structure_batch_list->resize(batch);

  for (int i_batch = 0; i_batch < batch; ++i_batch) {
    SingleBatchPostprocessor(
        structure_probs_data, bbox_preds_data, slice_dim, prob_dim, box_dim,
        batch_det_img_info[i_batch][0], batch_det_img_info[i_batch][1],
        batch_det_img_info[i_batch][2], batch_det_img_info[i_batch][3],
        batch_det_img_info[i_batch][4], batch_det_img_info[i_batch][5],
        &bbox_batch_list->at(i_batch), &structure_batch_list->at(i_batch));
    structure_probs_data = structure_probs_data + structure_probs_length;
    bbox_preds_data = bbox_preds_data + bbox_preds_length;
  }
  return true;
}

} // namespace ocr
} // namespace vision
} // namespace ultra_infer
