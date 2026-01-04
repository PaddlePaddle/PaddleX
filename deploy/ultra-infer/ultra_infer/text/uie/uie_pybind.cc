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

#include "ultra_infer/pybind/main.h"

namespace py = pybind11;

namespace ultra_infer {

void BindUIE(pybind11::module &m) {
  py::class_<text::SchemaNode>(m, "SchemaNode")
      .def(py::init<>())
      .def(py::init<std::string, std::vector<text::SchemaNode>>(),
           py::arg("name"), py::arg("children"))
      .def_readwrite("name", &text::SchemaNode::name_)
      .def_readwrite("prefix", &text::SchemaNode::prefix_)
      .def_readwrite("relations", &text::SchemaNode::relations_)
      .def_readwrite("children", &text::SchemaNode::children_);

  py::enum_<text::SchemaLanguage>(m, "SchemaLanguage", py::arithmetic(),
                                  "The language of schema.")
      .value("ZH", text::SchemaLanguage::ZH)
      .value("EN", text::SchemaLanguage::EN);

  py::class_<text::UIEModel, UltraInferModel>(m, "UIEModel")
      .def(py::init<std::string, std::string, std::string, float, size_t,
                    std::vector<std::string>, int, RuntimeOption, ModelFormat,
                    text::SchemaLanguage>(),
           py::arg("model_file"), py::arg("params_file"), py::arg("vocab_file"),
           py::arg("position_prob"), py::arg("max_length"), py::arg("schema"),
           py::arg("batch_size"),
           py::arg("custom_option") = ultra_infer::RuntimeOption(),
           py::arg("model_format") = ultra_infer::ModelFormat::PADDLE,
           py::arg("schema_language") = text::SchemaLanguage::ZH)
      .def(py::init<std::string, std::string, std::string, float, size_t,
                    std::vector<text::SchemaNode>, int, RuntimeOption,
                    ModelFormat, text::SchemaLanguage>(),
           py::arg("model_file"), py::arg("params_file"), py::arg("vocab_file"),
           py::arg("position_prob"), py::arg("max_length"), py::arg("schema"),
           py::arg("batch_size"),
           py::arg("custom_option") = ultra_infer::RuntimeOption(),
           py::arg("model_format") = ultra_infer::ModelFormat::PADDLE,
           py::arg("schema_language") = text::SchemaLanguage::ZH)
      .def(py::init<std::string, std::string, std::string, float, size_t,
                    text::SchemaNode, int, RuntimeOption, ModelFormat,
                    text::SchemaLanguage>(),
           py::arg("model_file"), py::arg("params_file"), py::arg("vocab_file"),
           py::arg("position_prob"), py::arg("max_length"), py::arg("schema"),
           py::arg("batch_size"),
           py::arg("custom_option") = ultra_infer::RuntimeOption(),
           py::arg("model_format") = ultra_infer::ModelFormat::PADDLE,
           py::arg("schema_language") = text::SchemaLanguage::ZH)
      .def("set_schema",
           static_cast<void (text::UIEModel::*)(
               const std::vector<std::string> &)>(&text::UIEModel::SetSchema),
           py::arg("schema"))
      .def("set_schema",
           static_cast<void (text::UIEModel::*)(
               const std::vector<text::SchemaNode> &)>(
               &text::UIEModel::SetSchema),
           py::arg("schema"))
      .def("set_schema",
           static_cast<void (text::UIEModel::*)(const text::SchemaNode &)>(
               &text::UIEModel::SetSchema),
           py::arg("schema"))
      .def(
          "predict",
          [](text::UIEModel &self, const std::vector<std::string> &texts) {
            std::vector<
                std::unordered_map<std::string, std::vector<text::UIEResult>>>
                results;
            self.Predict(texts, &results);
            return results;
          },
          py::arg("text"));
}

} // namespace ultra_infer
