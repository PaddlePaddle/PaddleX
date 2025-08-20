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

#include "ultra_infer/text/uie/model.h"

#include <algorithm>
#include <codecvt>
#include <locale>
#include <queue>
#include <sstream>

#include "fast_tokenizer/pretokenizers/pretokenizer.h"
#include "fast_tokenizer/utils/utf8.h"
#include "ultra_infer/function/concat.h"
#include "ultra_infer/function/split.h"

namespace ultra_infer {
namespace text {

static std::string DBC2SBC(const std::string &content) {
  std::string result;
  size_t content_utf8_len = 0;
  while (content_utf8_len < content.length()) {
    uint32_t content_char;
    auto content_char_width = fast_tokenizer::utils::UTF8ToUInt32(
        content.data() + content_utf8_len, &content_char);
    content_char = fast_tokenizer::utils::UTF8ToUnicode(content_char);
    if (content_char == 0x3000) {
      content_char = 0x0020;
    } else {
      content_char -= 0xfee0;
    }
    if (!(content_char >= 0x0021 && content_char <= 0x7e)) {
      result.append(content.data() + content_utf8_len, content_char_width);
    } else {
      char dst_char[5] = {0};
      uint32_t utf8_uint32 = fast_tokenizer::utils::UnicodeToUTF8(content_char);
      uint32_t utf8_char_count =
          fast_tokenizer::utils::UnicodeToUTF8Char(utf8_uint32, dst_char);
      result.append(dst_char, utf8_char_count);
    }
    content_utf8_len += content_char_width;
  }
  return result;
}

static std::ostream &PrintResult(std::ostream &os, const UIEResult &result,
                                 int tab_size) {
  constexpr int TAB_OFFSET = 4;
  // Print text
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "text: " << result.text_ << "\n";

  // Print probability
  for (int i = 0; i < tab_size; ++i) {
    os << " ";
  }
  os << "probability: " << result.probability_ << "\n";

  if (result.start_ != 0 || result.end_ != 0) {
    // Print start
    for (int i = 0; i < tab_size; ++i) {
      os << " ";
    }
    os << "start: " << result.start_ << "\n";

    // Print end
    for (int i = 0; i < tab_size; ++i) {
      os << " ";
    }
    os << "end: " << result.end_ << "\n";
  }

  // Print relation
  if (result.relation_.size() > 0) {
    for (int i = 0; i < tab_size; ++i) {
      os << " ";
    }
    os << "relation:\n";
    for (auto &&curr_relation : result.relation_) {
      for (int i = 0; i < tab_size + TAB_OFFSET; ++i) {
        os << " ";
      }
      os << curr_relation.first << ":\n";
      for (int i = 0; i < curr_relation.second.size(); ++i) {
        PrintResult(os, curr_relation.second[i],
                    tab_size + TAB_OFFSET + TAB_OFFSET);
      }
    }
  }
  os << "\n";
  return os;
}

std::ostream &operator<<(std::ostream &os, const UIEResult &result) {
  return PrintResult(os, result, 0);
}

std::ostream &operator<<(
    std::ostream &os,
    const std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
        &results) {
  os << "The result:\n";
  for (int i = 0; i < results.size(); ++i) {
    for (auto &&curr_result : results[i]) {
      os << curr_result.first << ": \n";
      for (auto &&uie_result : curr_result.second) {
        PrintResult(os, uie_result, 4);
      }
    }
    os << std::endl;
  }
  return os;
}

std::string UIEResult::Str() const {
  std::ostringstream oss;
  oss << *this;
  return oss.str();
}

void Schema::CreateRoot(const std::string &name) {
  root_ = ultra_infer::utils::make_unique<SchemaNode>(name);
}

Schema::Schema(const std::string &schema, const std::string &name) {
  CreateRoot(name);
  root_->AddChild(schema);
}

Schema::Schema(const std::vector<std::string> &schema_list,
               const std::string &name) {
  CreateRoot(name);
  for (const auto &schema : schema_list) {
    root_->AddChild(schema);
  }
}

Schema::Schema(const std::vector<SchemaNode> &schema_list,
               const std::string &name) {
  CreateRoot(name);
  for (const auto &schema : schema_list) {
    root_->AddChild(schema);
  }
}

Schema::Schema(const SchemaNode &schema, const std::string &name) {
  CreateRoot(name);
  root_->AddChild(schema);
}

UIEModel::UIEModel(const std::string &model_file,
                   const std::string &params_file,
                   const std::string &vocab_file, float position_prob,
                   size_t max_length, const std::vector<std::string> &schema,
                   int batch_size,
                   const ultra_infer::RuntimeOption &custom_option,
                   const ultra_infer::ModelFormat &model_format,
                   SchemaLanguage schema_language)
    : max_length_(max_length), position_prob_(position_prob),
      schema_language_(schema_language), batch_size_(batch_size),
      tokenizer_(vocab_file) {
  runtime_option = custom_option;
  runtime_option.SetModelPath(model_file, params_file, model_format);
  initialized = Initialize();
  SetSchema(schema);
  tokenizer_.EnableTruncMethod(
      max_length, 0, fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);
}

UIEModel::UIEModel(const std::string &model_file,
                   const std::string &params_file,
                   const std::string &vocab_file, float position_prob,
                   size_t max_length, const std::vector<SchemaNode> &schema,
                   int batch_size,
                   const ultra_infer::RuntimeOption &custom_option,
                   const ultra_infer::ModelFormat &model_format,
                   SchemaLanguage schema_language)
    : max_length_(max_length), position_prob_(position_prob),
      schema_language_(schema_language), batch_size_(batch_size),
      tokenizer_(vocab_file) {
  runtime_option = custom_option;
  runtime_option.SetModelPath(model_file, params_file, model_format);
  initialized = Initialize();
  SetSchema(schema);
  tokenizer_.EnableTruncMethod(
      max_length, 0, fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);
}

UIEModel::UIEModel(const std::string &model_file,
                   const std::string &params_file,
                   const std::string &vocab_file, float position_prob,
                   size_t max_length, const SchemaNode &schema, int batch_size,
                   const ultra_infer::RuntimeOption &custom_option,
                   const ultra_infer::ModelFormat &model_format,
                   SchemaLanguage schema_language)
    : max_length_(max_length), position_prob_(position_prob),
      schema_language_(schema_language), batch_size_(batch_size),
      tokenizer_(vocab_file) {
  runtime_option = custom_option;
  runtime_option.SetModelPath(model_file, params_file, model_format);
  initialized = Initialize();
  SetSchema(schema);
  tokenizer_.EnableTruncMethod(
      max_length, 0, fast_tokenizer::core::Direction::RIGHT,
      fast_tokenizer::core::TruncStrategy::LONGEST_FIRST);
}

bool UIEModel::Initialize() {
  SetValidBackend();
  return InitRuntime();
}

void UIEModel::SetValidBackend() {
  // TODO(zhoushunjie): Add lite backend in future
  valid_cpu_backends = {Backend::ORT, Backend::OPENVINO, Backend::PDINFER,
                        Backend::LITE};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
}

void UIEModel::SetSchema(const std::vector<std::string> &schema) {
  schema_ = ultra_infer::utils::make_unique<Schema>(schema);
}

void UIEModel::SetSchema(const std::vector<SchemaNode> &schema) {
  schema_ = ultra_infer::utils::make_unique<Schema>(schema);
}

void UIEModel::SetSchema(const SchemaNode &schema) {
  schema_ = ultra_infer::utils::make_unique<Schema>(schema);
}

void UIEModel::AutoSplitter(const std::vector<std::string> &texts,
                            size_t max_length,
                            std::vector<std::string> *short_texts,
                            std::vector<std::vector<size_t>> *input_mapping) {
  size_t cnt_org = 0;
  size_t cnt_short = 0;
  for (auto &text : texts) {
    auto text_len = fast_tokenizer::utils::GetUnicodeLenFromUTF8(text.c_str(),
                                                                 text.length());
    if (text_len <= max_length) {
      short_texts->push_back(text);
      if (input_mapping->size() <= cnt_org) {
        input_mapping->push_back({cnt_short});
      } else {
        (*input_mapping)[cnt_org].push_back(cnt_short);
      }
      cnt_short += 1;
    } else {
      fast_tokenizer::pretokenizers::CharToBytesOffsetConverter converter(text);
      for (size_t start = 0; start < text_len; start += max_length) {
        size_t end = start + max_length;
        if (end > text_len) {
          end = text_len;
        }
        fast_tokenizer::core::Offset byte_offset;
        converter.convert({start, end}, &byte_offset);
        short_texts->emplace_back(text.data() + byte_offset.first,
                                  byte_offset.second - byte_offset.first);
      }
      auto short_idx = cnt_short;
      cnt_short += text_len / max_length;
      if (text_len % max_length != 0) {
        ++cnt_short;
      }
      std::vector<size_t> temp_text_id(cnt_short - short_idx);
      std::iota(temp_text_id.begin(), temp_text_id.end(), short_idx);
      if (input_mapping->size() <= cnt_org) {
        input_mapping->push_back(std::move(temp_text_id));
      } else {
        (*input_mapping)[cnt_org].insert((*input_mapping)[cnt_org].end(),
                                         temp_text_id.begin(),
                                         temp_text_id.end());
      }
    }
    cnt_org += 1;
  }
}

void UIEModel::GetCandidateIdx(
    const float *probs, int64_t batch_size, int64_t seq_len,
    std::vector<std::vector<std::pair<int64_t, float>>> *candidate_idx_prob,
    float threshold) const {
  for (int i = 0; i < batch_size; ++i) {
    candidate_idx_prob->push_back({});
    for (int j = 0; j < seq_len; ++j) {
      if (probs[i * seq_len + j] > threshold) {
        candidate_idx_prob->back().push_back({j, probs[i * seq_len + j]});
      }
    }
  }
}

bool UIEModel::IdxProbCmp::operator()(
    const std::pair<IDX_PROB, IDX_PROB> &lhs,
    const std::pair<IDX_PROB, IDX_PROB> &rhs) const {
  if (lhs.first.first == rhs.first.first) {
    return lhs.second.first < rhs.second.first;
  }
  return lhs.first.first < rhs.first.first;
}

void UIEModel::GetSpan(const std::vector<IDX_PROB> &start_idx_prob,
                       const std::vector<IDX_PROB> &end_idx_prob,
                       SPAN_SET *span_set) const {
  size_t start_pointer = 0;
  size_t end_pointer = 0;
  size_t len_start = start_idx_prob.size();
  size_t len_end = end_idx_prob.size();
  while (start_pointer < len_start && end_pointer < len_end) {
    if (start_idx_prob[start_pointer].first ==
        end_idx_prob[end_pointer].first) {
      span_set->insert(std::make_pair(start_idx_prob[start_pointer],
                                      end_idx_prob[end_pointer]));
      ++start_pointer;
      ++end_pointer;
    } else if (start_idx_prob[start_pointer].first <
               end_idx_prob[end_pointer].first) {
      span_set->insert(std::make_pair(start_idx_prob[start_pointer],
                                      end_idx_prob[end_pointer]));
      ++start_pointer;
    } else {
      ++end_pointer;
    }
  }
}
void UIEModel::GetSpanIdxAndProbs(
    const SPAN_SET &span_set,
    const std::vector<fast_tokenizer::core::Offset> &offset_mapping,
    std::vector<SpanIdx> *span_idxs, std::vector<float> *probs) const {
  auto first_sep_idx =
      std::find_if(offset_mapping.begin() + 1, offset_mapping.end(),
                   [](const fast_tokenizer::core::Offset &offset) {
                     return offset == fast_tokenizer::core::Offset(0, 0);
                   });
  auto prompt_end_token_id =
      std::distance(offset_mapping.begin(), first_sep_idx) - 1;
  for (auto &&span_item : span_set) {
    probs->push_back(span_item.first.second * span_item.second.second);
    auto start_id = offset_mapping[span_item.first.first].first;
    auto end_id = offset_mapping[span_item.second.first].second;
    bool is_prompt = span_item.second.first <= prompt_end_token_id &&
                     span_item.second.first > 0;
    span_idxs->push_back({{start_id, end_id}, is_prompt});
  }
}

void UIEModel::ConvertSpanToUIEResult(
    const std::vector<std::string> &texts,
    const std::vector<std::string> &prompts,
    const std::vector<std::vector<SpanIdx>> &span_idxs,
    const std::vector<std::vector<float>> &probs,
    std::vector<std::vector<UIEResult>> *results) const {
  auto batch_size = texts.size();
  for (int i = 0; i < batch_size; ++i) {
    std::vector<UIEResult> result_list;
    if (span_idxs[i].size() == 0) {
      results->push_back({});
      continue;
    }
    auto &&text = texts[i];
    auto &&prompt = prompts[i];
    for (int j = 0; j < span_idxs[i].size(); ++j) {
      auto start = span_idxs[i][j].offset_.first;
      auto end = span_idxs[i][j].offset_.second;
      std::string span_text;
      std::vector<uint32_t> offset_mapping;
      if (span_idxs[i][j].is_prompt_) {
        fast_tokenizer::pretokenizers::CharToBytesOffsetConverter converter(
            prompt);
        fast_tokenizer::core::Offset byte_offset;
        converter.convert({start, end}, &byte_offset);
        span_text = prompt.substr(byte_offset.first,
                                  byte_offset.second - byte_offset.first);
        // Indicate cls task
        start = 0;
        end = 0;
      } else {
        fast_tokenizer::pretokenizers::CharToBytesOffsetConverter converter(
            text);
        fast_tokenizer::core::Offset byte_offset;
        converter.convert({start, end}, &byte_offset);
        span_text = text.substr(byte_offset.first,
                                byte_offset.second - byte_offset.first);
      }
      result_list.emplace_back(start, end, probs[i][j], span_text);
    }
    results->push_back(result_list);
  }
}

void UIEModel::AutoJoiner(const std::vector<std::string> &short_texts,
                          const std::vector<std::vector<size_t>> &input_mapping,
                          std::vector<std::vector<UIEResult>> *results) {
  bool is_cls_task = false;
  // 1. Detect if it's a cls task
  for (auto &&short_result : *results) {
    if (short_result.size() == 0) {
      continue;
    } else if (short_result[0].start_ == 0 && short_result[0].end_ == 0) {
      is_cls_task = true;
      break;
    } else {
      break;
    }
  }
  // 2. Get the final result
  std::vector<std::vector<UIEResult>> final_result;
  if (is_cls_task) {
    for (auto &&input_mapping_item : input_mapping) {
      std::unordered_map<std::string, std::pair<int, float>> cls_options;
      for (auto &&result_idx : input_mapping_item) {
        if ((*results)[result_idx].size() == 0) {
          continue;
        }
        auto &&text = (*results)[result_idx].front().text_;
        auto &&probability = (*results)[result_idx].front().probability_;
        if (cls_options.count(text) == 0) {
          cls_options[text] = std::make_pair(1, probability);
        } else {
          cls_options[text].first += 1;
          cls_options[text].second += probability;
        }
      }
      std::vector<UIEResult> result_list;
      if (cls_options.size() > 0) {
        auto max_iter = std::max_element(
            cls_options.begin(), cls_options.end(),
            [](const std::pair<std::string, std::pair<int, float>> &lhs,
               const std::pair<std::string, std::pair<int, float>> &rhs) {
              return lhs.second.second < rhs.second.second;
            });
        result_list.emplace_back(
            0, 0, max_iter->second.second / max_iter->second.first,
            max_iter->first);
      }
      final_result.push_back(result_list);
    }
  } else {
    for (auto &&input_mapping_item : input_mapping) {
      size_t offset = 0;
      std::vector<UIEResult> result_list;
      for (auto &&result_idx : input_mapping_item) {
        if (result_idx == 0) {
          result_list = std::move((*results)[result_idx]);
          offset += fast_tokenizer::utils::GetUnicodeLenFromUTF8(
              short_texts[result_idx].c_str(), short_texts[result_idx].size());
        } else {
          for (auto &&curr_result : (*results)[result_idx]) {
            curr_result.start_ += offset;
            curr_result.end_ += offset;
          }
          offset += fast_tokenizer::utils::GetUnicodeLenFromUTF8(
              short_texts[result_idx].c_str(), short_texts[result_idx].size());
          result_list.insert(result_list.end(), (*results)[result_idx].begin(),
                             (*results)[result_idx].end());
        }
      }
      final_result.push_back(result_list);
    }
  }
  *results = std::move(final_result);
}

bool UIEModel::ConstructTextsAndPrompts(
    const std::vector<std::string> &raw_texts, const std::string &node_name,
    const std::vector<std::vector<std::string>> node_prefix,
    std::vector<std::string> *input_texts, std::vector<std::string> *prompts,
    std::vector<std::vector<size_t>> *input_mapping_with_raw_texts,
    std::vector<std::vector<size_t>> *input_mapping) {
  size_t idx = 0;
  if (node_prefix.empty()) {
    for (int i = 0; i < raw_texts.size(); ++i) {
      input_texts->push_back(raw_texts[i]);
      prompts->push_back(DBC2SBC(node_name));
      input_mapping_with_raw_texts->push_back({idx});
      idx += 1;
    }
  } else {
    for (int i = 0; i < raw_texts.size(); ++i) {
      if (node_prefix[i].size() == 0) {
        input_mapping_with_raw_texts->push_back({});
      } else {
        for (auto &&pre : node_prefix[i]) {
          input_texts->push_back(raw_texts[i]);
          prompts->push_back(DBC2SBC(pre + node_name));
        }
        auto prefix_len = node_prefix[i].size();
        input_mapping_with_raw_texts->push_back({});
        input_mapping_with_raw_texts->back().resize(prefix_len);
        std::iota(input_mapping_with_raw_texts->back().begin(),
                  input_mapping_with_raw_texts->back().end(), idx);
        idx += prefix_len;
      }
    }
  }

  if (prompts->size() == 0) {
    return false;
  }

  // Shortten the input texts and prompts
  auto max_prompt_iter = std::max_element(
      prompts->begin(), prompts->end(),
      [](const std::string &lhs, const std::string &rhs) {
        auto lhs_ulen = fast_tokenizer::utils::GetUnicodeLenFromUTF8(
            lhs.c_str(), lhs.length());
        auto rhs_ulen = fast_tokenizer::utils::GetUnicodeLenFromUTF8(
            rhs.c_str(), rhs.length());
        return lhs_ulen < rhs_ulen;
      });
  auto max_prompt_len = fast_tokenizer::utils::GetUnicodeLenFromUTF8(
      max_prompt_iter->c_str(), max_prompt_iter->length());
  auto max_predict_len = max_length_ - 3 - max_prompt_len;

  std::vector<std::string> short_texts;
  AutoSplitter(*input_texts, max_predict_len, &short_texts, input_mapping);

  std::vector<std::string> short_texts_prompts;
  for (int i = 0; i < input_mapping->size(); ++i) {
    short_texts_prompts.insert(short_texts_prompts.end(),
                               (*input_mapping)[i].size(), (*prompts)[i]);
  }
  (*input_texts) = std::move(short_texts);
  (*prompts) = std::move(short_texts_prompts);
  return true;
}

void UIEModel::Preprocess(
    const std::vector<std::string> &input_texts,
    const std::vector<std::string> &prompts,
    std::vector<fast_tokenizer::core::Encoding> *encodings,
    std::vector<ultra_infer::FDTensor> *inputs) {
  // 1. Tokenize the short texts and short prompts
  std::vector<fast_tokenizer::core::EncodeInput> text_pair_input;
  for (int i = 0; i < input_texts.size(); ++i) {
    text_pair_input.emplace_back(
        std::pair<std::string, std::string>(prompts[i], input_texts[i]));
  }
  tokenizer_.EncodeBatchStrings(text_pair_input, encodings);
  // 2. Construct the input vector tensor
  // 2.1 Allocate input tensor
  int64_t batch_size = input_texts.size();
  int64_t seq_len = 0;
  if (batch_size > 0) {
    seq_len = (*encodings)[0].GetIds().size();
  }
  inputs->resize(NumInputsOfRuntime());
  for (int i = 0; i < NumInputsOfRuntime(); ++i) {
    (*inputs)[i].Allocate({batch_size, seq_len}, ultra_infer::FDDataType::INT64,
                          InputInfoOfRuntime(i).name);
  }

  // 2.2 Set the value of data
  size_t start = 0;
  int64_t *input_ids_ptr =
      reinterpret_cast<int64_t *>((*inputs)[0].MutableData());
  int64_t *type_ids_ptr =
      reinterpret_cast<int64_t *>((*inputs)[1].MutableData());
  int64_t *pos_ids_ptr =
      reinterpret_cast<int64_t *>((*inputs)[2].MutableData());
  int64_t *attn_mask_ptr =
      reinterpret_cast<int64_t *>((*inputs)[3].MutableData());

  for (int i = 0; i < encodings->size(); ++i) {
    auto &&curr_input_ids = (*encodings)[i].GetIds();
    auto &&curr_type_ids = (*encodings)[i].GetTypeIds();
    auto &&curr_attn_mask = (*encodings)[i].GetAttentionMask();

    std::copy(curr_input_ids.begin(), curr_input_ids.end(),
              input_ids_ptr + start);
    std::copy(curr_type_ids.begin(), curr_type_ids.end(), type_ids_ptr + start);
    std::iota(pos_ids_ptr + start, pos_ids_ptr + start + seq_len, 0);
    std::copy(curr_attn_mask.begin(), curr_attn_mask.end(),
              attn_mask_ptr + start);
    start += seq_len;
  }
}

void UIEModel::Postprocess(
    const std::vector<ultra_infer::FDTensor> &outputs,
    const std::vector<fast_tokenizer::core::Encoding> &encodings,
    const std::vector<std::string> &short_input_texts,
    const std::vector<std::string> &short_prompts,
    const std::vector<std::vector<size_t>> &input_mapping_with_short_text,
    std::vector<std::vector<UIEResult>> *results) {
  auto *start_prob = reinterpret_cast<const float *>(outputs[0].Data());
  auto *end_prob = reinterpret_cast<const float *>(outputs[1].Data());

  std::vector<std::vector<std::pair<int64_t, float>>> start_candidate_idx_prob,
      end_candidate_idx_prob;
  GetCandidateIdx(start_prob, outputs[0].shape[0], outputs[0].shape[1],
                  &start_candidate_idx_prob, position_prob_);
  GetCandidateIdx(end_prob, outputs[1].shape[0], outputs[1].shape[1],
                  &end_candidate_idx_prob, position_prob_);

  std::vector<std::vector<fast_tokenizer::core::Offset>> offset_mapping;
  for (int i = 0; i < encodings.size(); ++i) {
    auto &&curr_offsets = encodings[i].GetOffsets();
    offset_mapping.push_back(curr_offsets);
  }

  SPAN_SET span_set;
  auto batch_size = outputs[0].shape[0];
  std::vector<std::vector<float>> probs(batch_size);
  std::vector<std::vector<SpanIdx>> span_idxs(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    GetSpan(start_candidate_idx_prob[i], end_candidate_idx_prob[i], &span_set);
    GetSpanIdxAndProbs(span_set, offset_mapping[i], &span_idxs[i], &probs[i]);
    span_set.clear();
  }
  ConvertSpanToUIEResult(short_input_texts, short_prompts, span_idxs, probs,
                         results);
  AutoJoiner(short_input_texts, input_mapping_with_short_text, results);
}

void UIEModel::ConstructChildPromptPrefix(
    const std::vector<std::vector<size_t>> &input_mapping_with_raw_texts,
    const std::vector<std::vector<UIEResult>> &results_list,
    std::vector<std::vector<std::string>> *prefix) {
  prefix->resize(input_mapping_with_raw_texts.size());
  for (int i = 0; i < input_mapping_with_raw_texts.size(); ++i) {
    auto &&input_mapping_item = input_mapping_with_raw_texts[i];
    for (auto &&idx : input_mapping_item) {
      for (int j = 0; j < results_list[idx].size(); ++j) {
        std::string prefix_str;
        if (schema_language_ == SchemaLanguage::ZH) {
          // Note(zhoushunjie): It means "of" in Chinese.
          prefix_str = results_list[idx][j].text_ + "\xe7\x9a\x84";
        } else {
          prefix_str = " of " + results_list[idx][j].text_;
        }
        (*prefix)[i].push_back(prefix_str);
      }
    }
  }
}

void UIEModel::ConstructChildRelations(
    const std::vector<std::vector<UIEResult *>> &old_relations,
    const std::vector<std::vector<size_t>> &input_mapping_with_raw_texts,
    const std::vector<std::vector<UIEResult>> &results_list,
    const std::string &node_name,
    std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
        *results,
    std::vector<std::vector<UIEResult *>> *new_relations) {
  new_relations->resize(input_mapping_with_raw_texts.size());
  if (old_relations.size() == 0) {
    for (int i = 0; i < input_mapping_with_raw_texts.size(); ++i) {
      auto &&input_mapping_item = input_mapping_with_raw_texts[i];
      auto &curr_result = (*results)[i];
      for (auto &&idx : input_mapping_item) {
        if (results_list[idx].size() == 0) {
          continue;
        }
        if (curr_result.count(node_name) == 0) {
          curr_result[node_name] = results_list[idx];
        } else {
          curr_result[node_name].insert(curr_result[node_name].end(),
                                        results_list[idx].begin(),
                                        results_list[idx].end());
        }
      }
      if (curr_result.count(node_name) > 0) {
        for (auto &&curr_result_ref : curr_result[node_name]) {
          (*new_relations)[i].push_back(&curr_result_ref);
        }
      }
    }
  } else {
    auto &curr_relations = old_relations;
    for (int i = 0; i < input_mapping_with_raw_texts.size(); ++i) {
      auto &&input_mapping_item = input_mapping_with_raw_texts[i];
      for (int j = 0; j < input_mapping_item.size(); ++j) {
        auto idx = input_mapping_item[j];
        if (results_list[idx].size() == 0) {
          continue;
        }
        if (curr_relations[i][j]->relation_.count(node_name) == 0) {
          curr_relations[i][j]->relation_[node_name] = results_list[idx];
        } else {
          auto &curr_result = curr_relations[i][j]->relation_[node_name];
          curr_result.insert(curr_result.end(), results_list[idx].begin(),
                             results_list[idx].end());
        }
      }
    }
    for (int i = 0; i < curr_relations.size(); ++i) {
      for (int j = 0; j < curr_relations[i].size(); ++j) {
        if (curr_relations[i][j]->relation_.count(node_name)) {
          auto &curr_relation = curr_relations[i][j]->relation_[node_name];
          for (auto &&curr_result_ref : curr_relation) {
            (*new_relations)[i].push_back(&curr_result_ref);
          }
        }
      }
    }
  }
}

void UIEModel::Predict(
    const std::vector<std::string> &texts,
    std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
        *results) {
  std::queue<SchemaNode> nodes;
  for (auto &node : schema_->root_->children_) {
    nodes.push(node);
  }
  results->resize(texts.size());
  while (!nodes.empty()) {
    auto node = nodes.front();
    nodes.pop();
    std::vector<std::vector<size_t>> input_mapping_with_raw_texts;
    std::vector<std::vector<size_t>> input_mapping_with_short_text;
    std::vector<std::string> short_input_texts;
    std::vector<std::string> short_prompts;
    // 1. Construct texts and prompts from raw text
    bool has_prompt = ConstructTextsAndPrompts(
        texts, node.name_, node.prefix_, &short_input_texts, &short_prompts,
        &input_mapping_with_raw_texts, &input_mapping_with_short_text);
    std::vector<std::vector<UIEResult>> results_list;
    if (has_prompt) {
      // 2. Convert texts and prompts to FDTensor
      std::vector<FDTensor> inputs;
      std::vector<fast_tokenizer::core::Encoding> encodings;
      Preprocess(short_input_texts, short_prompts, &encodings, &inputs);

      std::vector<std::vector<FDTensor>> inputs_vec(NumInputsOfRuntime());
      int encoding_size = encodings.size();
      std::vector<int> num_or_sections;
      for (int i = 0; i < encoding_size; i += batch_size_) {
        int actual_batch_size = (std::min)(batch_size_, encoding_size - i);
        num_or_sections.push_back(actual_batch_size);
      }
      for (int i = 0; i < NumInputsOfRuntime(); ++i) {
        function::Split(inputs[i], num_or_sections, &inputs_vec[i]);
      }

      // 3. Infer
      std::vector<ultra_infer::FDTensor> outputs(NumOutputsOfRuntime());
      std::vector<ultra_infer::FDTensor> outputs0, outputs1;

      for (int i = 0; i < inputs_vec[0].size(); ++i) {
        std::vector<ultra_infer::FDTensor> curr_inputs(NumInputsOfRuntime());
        std::vector<ultra_infer::FDTensor> curr_outputs(NumOutputsOfRuntime());
        for (int j = 0; j < NumInputsOfRuntime(); ++j) {
          curr_inputs[j] = std::move(inputs_vec[j][i]);
          curr_inputs[j].name = inputs[j].name;
        }
        if (!Infer(curr_inputs, &curr_outputs)) {
          FDERROR << "Failed to inference while using model:" << ModelName()
                  << "." << std::endl;
        }
        outputs0.push_back(curr_outputs[0]);
        outputs1.push_back(curr_outputs[1]);
      }
      function::Concat(outputs0, &outputs[0]);
      function::Concat(outputs1, &outputs[1]);
      // 4. Convert FDTensor to UIEResult
      Postprocess(outputs, encodings, short_input_texts, short_prompts,
                  input_mapping_with_short_text, &results_list);
    }
    // 5. Construct the new relation of the UIEResult
    std::vector<std::vector<UIEResult *>> relations;
    ConstructChildRelations(node.relations_, input_mapping_with_raw_texts,
                            results_list, node.name_, results, &relations);

    // 6. Construct the next prompt prefix
    std::vector<std::vector<std::string>> prefix(texts.size());
    ConstructChildPromptPrefix(input_mapping_with_raw_texts, results_list,
                               &prefix);
    for (auto &node_child : node.children_) {
      node_child.relations_ = relations;
      node_child.prefix_ = prefix;
      nodes.push(node_child);
    }
  }
}

} // namespace text
} // namespace ultra_infer
