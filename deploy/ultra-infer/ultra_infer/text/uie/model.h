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

#pragma once

#include "fast_tokenizer/tokenizers/ernie_fast_tokenizer.h"
#include "ultra_infer/ultra_infer_model.h"
#include "ultra_infer/utils/unique_ptr.h"
#include <ostream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

using namespace paddlenlp;

namespace ultra_infer {
namespace text {

struct ULTRAINFER_DECL UIEResult {
  size_t start_;
  size_t end_;
  double probability_;
  std::string text_;
  std::unordered_map<std::string, std::vector<UIEResult>> relation_;
  UIEResult() = default;
  UIEResult(size_t start, size_t end, double probability, std::string text)
      : start_(start), end_(end), probability_(probability), text_(text) {}
  std::string Str() const;
};

ULTRAINFER_DECL std::ostream &operator<<(std::ostream &os,
                                         const UIEResult &result);
ULTRAINFER_DECL std::ostream &operator<<(
    std::ostream &os,
    const std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
        &results);

struct ULTRAINFER_DECL SchemaNode {
  std::string name_;
  std::vector<std::vector<std::string>> prefix_;
  std::vector<std::vector<UIEResult *>> relations_;
  std::vector<SchemaNode> children_;
  SchemaNode() = default;
  SchemaNode(const SchemaNode &) = default;
  explicit SchemaNode(const std::string &name,
                      const std::vector<SchemaNode> &children = {})
      : name_(name), children_(children) {}
  void AddChild(const std::string &schema) { children_.emplace_back(schema); }
  void AddChild(const SchemaNode &schema) { children_.push_back(schema); }
  void AddChild(const std::string &schema,
                const std::vector<std::string> &children) {
    SchemaNode schema_node(schema);
    for (auto &child : children) {
      schema_node.children_.emplace_back(child);
    }
    children_.emplace_back(schema_node);
  }
  void AddChild(const std::string &schema,
                const std::vector<SchemaNode> &children) {
    SchemaNode schema_node(schema);
    schema_node.children_ = children;
    children_.emplace_back(schema_node);
  }
};

enum SchemaLanguage {
  ZH, // Chinese
  EN  // English
};

struct Schema {
  explicit Schema(const std::string &schema, const std::string &name = "root");
  explicit Schema(const std::vector<std::string> &schema_list,
                  const std::string &name = "root");
  explicit Schema(const std::vector<SchemaNode> &schema_list,
                  const std::string &name = "root");
  explicit Schema(const SchemaNode &schema, const std::string &name = "root");

private:
  void CreateRoot(const std::string &name);
  std::unique_ptr<SchemaNode> root_;
  friend class UIEModel;
};

struct ULTRAINFER_DECL UIEModel : public UltraInferModel {
public:
  UIEModel(const std::string &model_file, const std::string &params_file,
           const std::string &vocab_file, float position_prob,
           size_t max_length, const std::vector<std::string> &schema,
           int batch_size,
           const ultra_infer::RuntimeOption &custom_option =
               ultra_infer::RuntimeOption(),
           const ultra_infer::ModelFormat &model_format =
               ultra_infer::ModelFormat::PADDLE,
           SchemaLanguage schema_language = SchemaLanguage::ZH);
  UIEModel(const std::string &model_file, const std::string &params_file,
           const std::string &vocab_file, float position_prob,
           size_t max_length, const SchemaNode &schema, int batch_size,
           const ultra_infer::RuntimeOption &custom_option =
               ultra_infer::RuntimeOption(),
           const ultra_infer::ModelFormat &model_format =
               ultra_infer::ModelFormat::PADDLE,
           SchemaLanguage schema_language = SchemaLanguage::ZH);
  UIEModel(const std::string &model_file, const std::string &params_file,
           const std::string &vocab_file, float position_prob,
           size_t max_length, const std::vector<SchemaNode> &schema,
           int batch_size,
           const ultra_infer::RuntimeOption &custom_option =
               ultra_infer::RuntimeOption(),
           const ultra_infer::ModelFormat &model_format =
               ultra_infer::ModelFormat::PADDLE,
           SchemaLanguage schema_language = SchemaLanguage::ZH);
  virtual std::string ModelName() const { return "UIEModel"; }
  void SetSchema(const std::vector<std::string> &schema);
  void SetSchema(const std::vector<SchemaNode> &schema);
  void SetSchema(const SchemaNode &schema);

  bool ConstructTextsAndPrompts(
      const std::vector<std::string> &raw_texts, const std::string &node_name,
      const std::vector<std::vector<std::string>> node_prefix,
      std::vector<std::string> *input_texts, std::vector<std::string> *prompts,
      std::vector<std::vector<size_t>> *input_mapping_with_raw_texts,
      std::vector<std::vector<size_t>> *input_mapping_with_short_text);
  void Preprocess(const std::vector<std::string> &input_texts,
                  const std::vector<std::string> &prompts,
                  std::vector<fast_tokenizer::core::Encoding> *encodings,
                  std::vector<ultra_infer::FDTensor> *inputs);
  void Postprocess(
      const std::vector<ultra_infer::FDTensor> &outputs,
      const std::vector<fast_tokenizer::core::Encoding> &encodings,
      const std::vector<std::string> &short_input_texts,
      const std::vector<std::string> &short_prompts,
      const std::vector<std::vector<size_t>> &input_mapping_with_short_text,
      std::vector<std::vector<UIEResult>> *results);
  void ConstructChildPromptPrefix(
      const std::vector<std::vector<size_t>> &input_mapping_with_raw_texts,
      const std::vector<std::vector<UIEResult>> &results_list,
      std::vector<std::vector<std::string>> *prefix);
  void ConstructChildRelations(
      const std::vector<std::vector<UIEResult *>> &old_relations,
      const std::vector<std::vector<size_t>> &input_mapping_with_raw_texts,
      const std::vector<std::vector<UIEResult>> &results_list,
      const std::string &node_name,
      std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
          *results,
      std::vector<std::vector<UIEResult *>> *new_relations);
  void
  Predict(const std::vector<std::string> &texts,
          std::vector<std::unordered_map<std::string, std::vector<UIEResult>>>
              *results);

protected:
  using IDX_PROB = std::pair<int64_t, float>;
  struct IdxProbCmp {
    bool operator()(const std::pair<IDX_PROB, IDX_PROB> &lhs,
                    const std::pair<IDX_PROB, IDX_PROB> &rhs) const;
  };
  using SPAN_SET = std::set<std::pair<IDX_PROB, IDX_PROB>, IdxProbCmp>;
  struct SpanIdx {
    fast_tokenizer::core::Offset offset_;
    bool is_prompt_;
  };
  void SetValidBackend();
  bool Initialize();
  void AutoSplitter(const std::vector<std::string> &texts, size_t max_length,
                    std::vector<std::string> *short_texts,
                    std::vector<std::vector<size_t>> *input_mapping);
  void AutoJoiner(const std::vector<std::string> &short_texts,
                  const std::vector<std::vector<size_t>> &input_mapping,
                  std::vector<std::vector<UIEResult>> *results);
  // Get idx of the last dimension in probability arrays, which is greater than
  // a limitation.
  void GetCandidateIdx(const float *probs, int64_t batch_size, int64_t seq_len,
                       std::vector<std::vector<IDX_PROB>> *candidate_idx_prob,
                       float threshold = 0.5) const;
  void GetSpan(const std::vector<IDX_PROB> &start_idx_prob,
               const std::vector<IDX_PROB> &end_idx_prob,
               SPAN_SET *span_set) const;
  void GetSpanIdxAndProbs(
      const SPAN_SET &span_set,
      const std::vector<fast_tokenizer::core::Offset> &offset_mapping,
      std::vector<SpanIdx> *span_idxs, std::vector<float> *probs) const;
  void
  ConvertSpanToUIEResult(const std::vector<std::string> &texts,
                         const std::vector<std::string> &prompts,
                         const std::vector<std::vector<SpanIdx>> &span_idxs,
                         const std::vector<std::vector<float>> &probs,
                         std::vector<std::vector<UIEResult>> *results) const;
  std::unique_ptr<Schema> schema_;
  size_t max_length_;
  float position_prob_;
  int batch_size_;
  SchemaLanguage schema_language_;
  fast_tokenizer::tokenizers_impl::ErnieFastTokenizer tokenizer_;
};

} // namespace text
} // namespace ultra_infer
